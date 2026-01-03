#!/usr/bin/env python3
"""
NeuroGen 2.0 - Encoder/Decoder Pre-training Script

This script pre-trains ONLY the encoder (TokenEmbedding) and decoder (GPUDecoder 
projection matrix) before main training. These are the only components whose weights
are saved and loaded - the SNN modules (Thalamus, Wernicke, Broca, etc.) learn
from scratch during main training.

Pre-training phases:
1. Decoder pre-training: Train projection matrix to map sparse Broca patterns -> vocabulary
2. Embedding pre-training: Skip-gram style co-occurrence learning for token embeddings

Why this matters:
- Random decoder projection gives near-uniform logits -> mode collapse
- Random embeddings have no semantic structure -> garbage in, garbage out
- Pre-training gives the SNN meaningful inputs and outputs to learn from

Usage:
    python pretrain_encoder_decoder.py --output checkpoints/pretrained.ckpt
    python pretrain_encoder_decoder.py --test  # Quick test mode
    python pretrain_encoder_decoder.py --intensive  # Maximum training
    
Then use the checkpoint in main training:
    python train_slimpajama.py --pretrained-checkpoint checkpoints/pretrained.ckpt
"""

import os
import sys
import time
import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from datasets import load_dataset
    import sentencepiece as spm
    from tqdm import tqdm
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except ImportError as e:
    print(f"Error: Missing required package - {e}")
    print("Install with: pip install datasets sentencepiece numpy tqdm matplotlib")
    sys.exit(1)


@dataclass
class PretrainConfig:
    """Pre-training configuration"""
    vocab_size: int = 32000
    embedding_dim: int = 768
    gpu_device: int = 0
    
    dataset_name: str = "cerebras/SlimPajama-627B"
    dataset_split: str = "train"
    streaming: bool = True
    max_seq_length: int = 256
    
    # Decoder pre-training (INTENSIVE)
    pretrain_decoder: bool = True
    decoder_iterations: int = 10000
    decoder_lr: float = 0.01
    decoder_lr_warmup: int = 500
    decoder_lr_decay: float = 0.95
    decoder_batch_size: int = 32
    decoder_sparsity: float = 0.10
    decoder_sparsity_var: float = 0.03
    
    # Embedding pre-training (INTENSIVE)
    pretrain_embeddings: bool = True
    embedding_sequences: int = 2000
    embedding_epochs: int = 3
    embedding_window: int = 5
    embedding_lr: float = 0.025
    embedding_lr_decay: float = 0.98
    embedding_negative_samples: int = 10
    embedding_subsample_threshold: float = 1e-4
    
    output_checkpoint: str = "checkpoints/pretrained.ckpt"
    tokenizer_dir: str = "tokenizer"
    tokenizer_model: str = "nlp_agent_tokenizer.model"
    viz_dir: str = "pretrain_viz"
    viz_enabled: bool = True
    viz_interval: int = 500


class PretrainVisualizer:
    """Visualizes pre-training progress"""
    
    def __init__(self, config: PretrainConfig):
        self.config = config
        self.viz_dir = Path(config.viz_dir)
        self.viz_dir.mkdir(exist_ok=True)
        
        self.decoder_losses: List[float] = []
        self.decoder_steps: List[int] = []
        self.decoder_lr_history: List[float] = []
        
        self.embedding_losses: List[float] = []
        self.embedding_steps: List[int] = []
        self.embedding_lr_history: List[float] = []
        
    def add_decoder_point(self, step: int, loss: float, lr: float):
        self.decoder_steps.append(step)
        self.decoder_losses.append(loss)
        self.decoder_lr_history.append(lr)
        
    def add_embedding_point(self, step: int, loss: float, lr: float):
        self.embedding_steps.append(step)
        self.embedding_losses.append(loss)
        self.embedding_lr_history.append(lr)
    
    def save_overview(self, phase: str = "both"):
        """Generate overview visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        ax = axes[0, 0]
        if self.decoder_losses:
            ax.plot(self.decoder_steps, self.decoder_losses, 'b-', alpha=0.7, linewidth=1)
            if len(self.decoder_losses) > 20:
                window = min(50, len(self.decoder_losses) // 5)
                smoothed = np.convolve(self.decoder_losses, np.ones(window)/window, mode='valid')
                smoothed_steps = self.decoder_steps[window-1:]
                ax.plot(smoothed_steps, smoothed, 'r-', linewidth=2, label=f'{window}-step MA')
                ax.legend()
            ax.set_title('Decoder Pre-training Loss', fontsize=12, fontweight='bold')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Cross-Entropy Loss')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, 'Not started', ha='center', va='center', fontsize=14)
            ax.set_title('Decoder Pre-training Loss')
        
        ax = axes[0, 1]
        if self.decoder_lr_history:
            ax.plot(self.decoder_steps, self.decoder_lr_history, 'g-', linewidth=2)
            ax.set_title('Decoder Learning Rate Schedule', fontsize=12, fontweight='bold')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Learning Rate')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Not started', ha='center', va='center', fontsize=14)
            ax.set_title('Decoder Learning Rate')
        
        ax = axes[1, 0]
        if self.embedding_losses:
            ax.plot(self.embedding_steps, self.embedding_losses, 'purple', alpha=0.7, linewidth=1)
            if len(self.embedding_losses) > 20:
                window = min(50, len(self.embedding_losses) // 5)
                smoothed = np.convolve(self.embedding_losses, np.ones(window)/window, mode='valid')
                smoothed_steps = self.embedding_steps[window-1:]
                ax.plot(smoothed_steps, smoothed, 'orange', linewidth=2, label=f'{window}-step MA')
                ax.legend()
            ax.set_title('Embedding Pre-training Loss (Skip-gram)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Step')
            ax.set_ylabel('Negative Sampling Loss')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Not started', ha='center', va='center', fontsize=14)
            ax.set_title('Embedding Pre-training Loss')
        
        ax = axes[1, 1]
        if self.embedding_lr_history:
            ax.plot(self.embedding_steps, self.embedding_lr_history, 'orange', linewidth=2)
            ax.set_title('Embedding Learning Rate Schedule', fontsize=12, fontweight='bold')
            ax.set_xlabel('Step')
            ax.set_ylabel('Learning Rate')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Not started', ha='center', va='center', fontsize=14)
            ax.set_title('Embedding Learning Rate')
        
        plt.suptitle('NeuroGen 2.0 Encoder/Decoder Pre-training', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.viz_dir / 'pretraining_progress.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    def save_final_summary(self, decoder_loss: float, embedding_loss: float, elapsed: float):
        """Generate final summary visualization"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        if self.decoder_losses:
            axes[0].plot(self.decoder_steps, self.decoder_losses, 'b-', alpha=0.5)
            if len(self.decoder_losses) > 10:
                window = min(50, len(self.decoder_losses) // 5)
                smoothed = np.convolve(self.decoder_losses, np.ones(window)/window, mode='valid')
                axes[0].plot(self.decoder_steps[window-1:], smoothed, 'r-', linewidth=2)
            axes[0].axhline(y=decoder_loss, color='green', linestyle='--', label=f'Final: {decoder_loss:.4f}')
            axes[0].legend()
            axes[0].set_title('Decoder Training', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('Loss')
            axes[0].grid(True, alpha=0.3)
        
        if self.embedding_losses:
            axes[1].plot(self.embedding_steps, self.embedding_losses, 'purple', alpha=0.5)
            if len(self.embedding_losses) > 10:
                window = min(50, len(self.embedding_losses) // 5)
                smoothed = np.convolve(self.embedding_losses, np.ones(window)/window, mode='valid')
                axes[1].plot(self.embedding_steps[window-1:], smoothed, 'orange', linewidth=2)
            axes[1].axhline(y=embedding_loss, color='green', linestyle='--', label=f'Final: {embedding_loss:.4f}')
            axes[1].legend()
            axes[1].set_title('Embedding Training (Skip-gram)', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Step')
            axes[1].set_ylabel('Loss')
            axes[1].grid(True, alpha=0.3)
        
        axes[2].axis('off')
        if self.decoder_losses and self.embedding_losses:
            dec_reduction = (self.decoder_losses[0] - decoder_loss) / self.decoder_losses[0] * 100
            emb_reduction = (self.embedding_losses[0] - embedding_loss) / self.embedding_losses[0] * 100
            summary_text = f"""
Pre-training Summary

Decoder Pre-training:
  Iterations: {len(self.decoder_losses)}
  Final Loss: {decoder_loss:.4f}
  Loss Reduction: {dec_reduction:.1f}%

Embedding Pre-training:
  Steps: {len(self.embedding_losses)}  
  Final Loss: {embedding_loss:.4f}
  Loss Reduction: {emb_reduction:.1f}%

Total Time: {elapsed:.1f}s ({elapsed/60:.1f} min)

Ready for main training!
            """
        else:
            summary_text = "Pre-training incomplete"
        
        axes[2].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                    verticalalignment='center', transform=axes[2].transAxes)
        
        plt.suptitle('NeuroGen 2.0 Pre-training Complete', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.viz_dir / 'pretraining_final.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved final summary: {output_path}")


class EncoderDecoderPretrainer:
    """Pre-trains the encoder (embeddings) and decoder (projection matrix) only."""
    
    def __init__(self, config: PretrainConfig):
        self.config = config
        self.start_time = time.time()
        
        print(f"Loading SentencePiece tokenizer from {config.tokenizer_dir}")
        tokenizer_path = Path(config.tokenizer_dir) / config.tokenizer_model
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(str(tokenizer_path))
        
        tokenizer_state_path = Path(config.tokenizer_dir) / "tokenizer_state.json"
        with open(tokenizer_state_path, 'r') as f:
            tokenizer_config = json.load(f)
        actual_vocab_size = tokenizer_config.get('vocab_size', 32000)
        print(f"Tokenizer loaded: vocab_size={actual_vocab_size}")
        
        if actual_vocab_size != config.vocab_size:
            print(f"Updating vocab_size from {config.vocab_size} to {actual_vocab_size}")
            config.vocab_size = actual_vocab_size
        
        sys.path.insert(0, str(Path(__file__).parent / "bin"))
        
        try:
            import libneurogen
            self.model = libneurogen.NeuroGenModel(
                vocab_size=config.vocab_size,
                embedding_dim=config.embedding_dim,
                gpu_device=config.gpu_device
            )
            print("Loaded libneurogen bindings")
        except ImportError as e:
            print(f"Failed to load libneurogen: {e}")
            print("Make sure the project is built with 'make'")
            sys.exit(1)
        
        if config.viz_enabled:
            self.viz = PretrainVisualizer(config)
        else:
            self.viz = None
        
        output_dir = Path(config.output_checkpoint).parent
        output_dir.mkdir(exist_ok=True, parents=True)
        
        self.token_counts: Dict[int, int] = {}
        self.total_tokens: int = 0
    
    def collect_sequences(self, num_sequences: int) -> Tuple[List[List[int]], Dict[int, int]]:
        """Collect token sequences and compute token frequencies."""
        print(f"\nCollecting {num_sequences} sequences from dataset...")
        
        try:
            dataset = load_dataset(
                self.config.dataset_name,
                split=self.config.dataset_split,
                streaming=self.config.streaming
            )
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            return [], {}
        
        sequences = []
        token_counts: Dict[int, int] = {}
        total_tokens = 0
        skipped = 0
        
        with tqdm(total=num_sequences, desc="Collecting sequences") as pbar:
            for i, example in enumerate(dataset):
                if len(sequences) >= num_sequences:
                    break
                
                text = example.get("text", "")
                if not text:
                    skipped += 1
                    continue
                
                tokens = self.tokenizer.EncodeAsIds(text)
                
                if len(tokens) >= 10:
                    seq = tokens[:self.config.max_seq_length]
                    sequences.append(seq)
                    
                    for token in seq:
                        token_counts[token] = token_counts.get(token, 0) + 1
                        total_tokens += 1
                    
                    pbar.update(1)
                else:
                    skipped += 1
        
        self.token_counts = token_counts
        self.total_tokens = total_tokens
        
        print(f"Collected {len(sequences)} sequences ({skipped} skipped)")
        print(f"   Total tokens: {total_tokens:,}")
        print(f"   Unique tokens: {len(token_counts):,}")
        
        return sequences, token_counts
    
    def get_learning_rate(self, step: int, base_lr: float, warmup_steps: int, 
                          decay_factor: float, decay_interval: int = 1000) -> float:
        """Compute learning rate with warmup and decay."""
        if step < warmup_steps:
            return base_lr * (step + 1) / warmup_steps
        else:
            decay_steps = (step - warmup_steps) // decay_interval
            return base_lr * (decay_factor ** decay_steps)
    
    def pretrain_decoder(self) -> float:
        """Pre-train the decoder projection matrix with intensive training."""
        print("\n" + "=" * 70)
        print("PHASE 1: INTENSIVE Decoder Pre-training")
        print("=" * 70)
        print(f"   Iterations:    {self.config.decoder_iterations}")
        print(f"   Base LR:       {self.config.decoder_lr}")
        print(f"   Warmup steps:  {self.config.decoder_lr_warmup}")
        print(f"   LR decay:      {self.config.decoder_lr_decay} per 1000 iters")
        print(f"   Batch size:    {self.config.decoder_batch_size}")
        print(f"   Sparsity:      {self.config.decoder_sparsity:.1%}")
        print("=" * 70)
        
        total_loss = 0.0
        num_updates = 0
        recent_losses = []
        
        try:
            with tqdm(total=self.config.decoder_iterations, desc="Decoder training") as pbar:
                for step in range(self.config.decoder_iterations):
                    lr = self.get_learning_rate(
                        step, 
                        self.config.decoder_lr,
                        self.config.decoder_lr_warmup,
                        self.config.decoder_lr_decay,
                        decay_interval=1000
                    )
                    
                    loss = self.model.pretrain_decoder(
                        num_iterations=1,
                        learning_rate=lr
                    )
                    
                    total_loss += loss
                    num_updates += 1
                    recent_losses.append(loss)
                    if len(recent_losses) > 100:
                        recent_losses.pop(0)
                    
                    if self.viz and step % 10 == 0:
                        self.viz.add_decoder_point(step, loss, lr)
                    
                    avg_recent = sum(recent_losses) / len(recent_losses)
                    pbar.set_postfix({
                        'loss': f'{loss:.4f}',
                        'avg': f'{avg_recent:.4f}',
                        'lr': f'{lr:.5f}'
                    })
                    pbar.update(1)
                    
                    if self.viz and step > 0 and step % self.config.viz_interval == 0:
                        self.viz.save_overview("decoder")
            
            final_loss = total_loss / max(num_updates, 1)
            
            if self.viz:
                self.viz.save_overview("decoder")
            
            print(f"\nDecoder pre-training complete")
            print(f"   Final average loss: {final_loss:.4f}")
            print(f"   Recent 100-step avg: {sum(recent_losses)/len(recent_losses):.4f}")
            return final_loss
            
        except Exception as e:
            print(f"Decoder pre-training failed: {e}")
            import traceback
            traceback.print_exc()
            return float('inf')
    
    def pretrain_embeddings(self, sequences: List[List[int]]) -> float:
        """Pre-train token embeddings using intensive Skip-gram training."""
        print("\n" + "=" * 70)
        print("PHASE 2: INTENSIVE Embedding Pre-training (Skip-gram)")
        print("=" * 70)
        print(f"   Sequences:         {len(sequences)}")
        print(f"   Epochs:            {self.config.embedding_epochs}")
        print(f"   Window size:       {self.config.embedding_window}")
        print(f"   Base LR:           {self.config.embedding_lr}")
        print(f"   LR decay/epoch:    {self.config.embedding_lr_decay}")
        print(f"   Negative samples:  {self.config.embedding_negative_samples}")
        print("=" * 70)
        
        if not sequences:
            print("No sequences provided, skipping embedding pre-training")
            return float('inf')
        
        total_loss = 0.0
        total_steps = 0
        recent_losses = []
        
        try:
            for epoch in range(self.config.embedding_epochs):
                epoch_lr = self.config.embedding_lr * (self.config.embedding_lr_decay ** epoch)
                
                print(f"\n--- Epoch {epoch + 1}/{self.config.embedding_epochs} (LR: {epoch_lr:.5f}) ---")
                
                epoch_sequences = sequences.copy()
                np.random.shuffle(epoch_sequences)
                
                epoch_loss = 0.0
                epoch_steps = 0
                
                with tqdm(total=len(epoch_sequences), desc=f"Epoch {epoch+1}") as pbar:
                    for seq_idx, sequence in enumerate(epoch_sequences):
                        loss = self.model.pretrain_embeddings(
                            token_sequences=[sequence],
                            window_size=self.config.embedding_window,
                            learning_rate=epoch_lr,
                            negative_samples=self.config.embedding_negative_samples
                        )
                        
                        epoch_loss += loss
                        epoch_steps += 1
                        total_loss += loss
                        total_steps += 1
                        recent_losses.append(loss)
                        if len(recent_losses) > 100:
                            recent_losses.pop(0)
                        
                        if self.viz and seq_idx % 10 == 0:
                            self.viz.add_embedding_point(total_steps, loss, epoch_lr)
                        
                        avg_recent = sum(recent_losses) / len(recent_losses)
                        pbar.set_postfix({
                            'loss': f'{loss:.4f}',
                            'avg': f'{avg_recent:.4f}'
                        })
                        pbar.update(1)
                
                epoch_avg = epoch_loss / max(epoch_steps, 1)
                print(f"   Epoch {epoch+1} average loss: {epoch_avg:.4f}")
                
                if self.viz:
                    self.viz.save_overview("embedding")
            
            final_loss = total_loss / max(total_steps, 1)
            
            print(f"\nEmbedding pre-training complete")
            print(f"   Final average loss: {final_loss:.4f}")
            print(f"   Total steps: {total_steps}")
            return final_loss
            
        except Exception as e:
            print(f"Embedding pre-training failed: {e}")
            import traceback
            traceback.print_exc()
            return float('inf')
    
    def run(self) -> Dict[str, Any]:
        """Run all pre-training phases (decoder and embeddings only)."""
        print("\n" + "=" * 80)
        print("NeuroGen 2.0 - ENCODER/DECODER PRE-TRAINING")
        print("=" * 80)
        print("This establishes meaningful representations before main training:")
        print("  1. Decoder:    Map sparse spike patterns -> vocabulary tokens")
        print("  2. Embeddings: Group similar tokens together (Skip-gram)")
        print("")
        print("NOTE: SNN modules (Thalamus, Wernicke, Broca, etc.) are NOT pre-trained")
        print("      because their weights are not saved. They learn during main training.")
        print("=" * 80 + "\n")
        
        results = {
            'decoder_loss': None,
            'embedding_loss': None,
            'total_time': 0.0
        }
        
        if self.config.pretrain_decoder:
            results['decoder_loss'] = self.pretrain_decoder()
        else:
            print("\nSkipping decoder pre-training (disabled)")
        
        sequences = []
        if self.config.pretrain_embeddings:
            sequences, _ = self.collect_sequences(self.config.embedding_sequences)
        
        if self.config.pretrain_embeddings:
            if sequences:
                results['embedding_loss'] = self.pretrain_embeddings(sequences)
            else:
                print("No sequences available, skipping embedding pre-training")
        else:
            print("\nSkipping embedding pre-training (disabled)")
        
        print("\n" + "=" * 70)
        print(f"Saving pre-trained weights to: {self.config.output_checkpoint}")
        print("=" * 70)
        
        try:
            self.model.save_checkpoint(self.config.output_checkpoint)
            print("Checkpoint saved successfully")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
            return results
        
        elapsed = time.time() - self.start_time
        results['total_time'] = elapsed
        
        if self.viz:
            decoder_loss = results['decoder_loss'] if results['decoder_loss'] is not None else 0.0
            embedding_loss = results['embedding_loss'] if results['embedding_loss'] is not None else 0.0
            self.viz.save_final_summary(decoder_loss, embedding_loss, elapsed)
        
        print("\n" + "=" * 80)
        print("PRE-TRAINING COMPLETE")
        print("=" * 80)
        print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        print(f"\nResults:")
        if results['decoder_loss'] is not None:
            print(f"  Decoder loss:   {results['decoder_loss']:.4f}")
        if results['embedding_loss'] is not None:
            print(f"  Embedding loss: {results['embedding_loss']:.4f}")
        print(f"\nCheckpoint saved to: {self.config.output_checkpoint}")
        print(f"\nNext step: Run main training with pre-trained weights:")
        print(f"   python train_slimpajama.py --pretrained-checkpoint {self.config.output_checkpoint}")
        print("=" * 80)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train NeuroGen encoder/decoder before main training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard pre-training
  python pretrain_encoder_decoder.py --output checkpoints/pretrained.ckpt
  
  # Intensive pre-training (recommended for best results)
  python pretrain_encoder_decoder.py --intensive
  
  # Quick test mode
  python pretrain_encoder_decoder.py --test
  
  # Only pre-train decoder
  python pretrain_encoder_decoder.py --no-embeddings
  
  # Custom settings
  python pretrain_encoder_decoder.py --decoder-iters 20000 --embed-epochs 5
        """
    )
    
    parser.add_argument("--output", type=str, default="checkpoints/pretrained.ckpt",
                        help="Output checkpoint path")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    
    parser.add_argument("--test", action="store_true",
                        help="Quick test mode with reduced iterations")
    parser.add_argument("--intensive", action="store_true",
                        help="Intensive training mode (2x iterations, more epochs)")
    
    parser.add_argument("--no-decoder", action="store_true",
                        help="Skip decoder pre-training")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Skip embedding pre-training")
    
    parser.add_argument("--decoder-iters", type=int, default=10000,
                        help="Decoder pre-training iterations (default: 10000)")
    parser.add_argument("--decoder-lr", type=float, default=0.01,
                        help="Decoder base learning rate (default: 0.01)")
    parser.add_argument("--decoder-warmup", type=int, default=500,
                        help="Decoder LR warmup steps (default: 500)")
    
    parser.add_argument("--embed-seqs", type=int, default=2000,
                        help="Number of sequences for embedding pre-training (default: 2000)")
    parser.add_argument("--embed-epochs", type=int, default=3,
                        help="Embedding pre-training epochs (default: 3)")
    parser.add_argument("--embed-window", type=int, default=5,
                        help="Skip-gram window size (default: 5)")
    parser.add_argument("--embed-lr", type=float, default=0.025,
                        help="Embedding base learning rate (default: 0.025)")
    
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable visualization")
    
    args = parser.parse_args()
    
    if args.test:
        print("Running in TEST MODE (reduced iterations)")
        config = PretrainConfig(
            gpu_device=args.gpu,
            output_checkpoint=args.output,
            pretrain_decoder=not args.no_decoder,
            decoder_iterations=500,
            decoder_lr=args.decoder_lr,
            decoder_lr_warmup=50,
            pretrain_embeddings=not args.no_embeddings,
            embedding_sequences=100,
            embedding_epochs=1,
            embedding_window=args.embed_window,
            embedding_lr=args.embed_lr,
            viz_enabled=not args.no_viz,
            viz_interval=50
        )
    elif args.intensive:
        print("Running in INTENSIVE MODE (maximum training)")
        config = PretrainConfig(
            gpu_device=args.gpu,
            output_checkpoint=args.output,
            pretrain_decoder=not args.no_decoder,
            decoder_iterations=20000,
            decoder_lr=args.decoder_lr,
            decoder_lr_warmup=1000,
            decoder_lr_decay=0.97,
            pretrain_embeddings=not args.no_embeddings,
            embedding_sequences=5000,
            embedding_epochs=5,
            embedding_window=args.embed_window,
            embedding_lr=args.embed_lr,
            embedding_negative_samples=15,
            viz_enabled=not args.no_viz,
            viz_interval=500
        )
    else:
        config = PretrainConfig(
            gpu_device=args.gpu,
            output_checkpoint=args.output,
            pretrain_decoder=not args.no_decoder,
            decoder_iterations=args.decoder_iters,
            decoder_lr=args.decoder_lr,
            decoder_lr_warmup=args.decoder_warmup,
            pretrain_embeddings=not args.no_embeddings,
            embedding_sequences=args.embed_seqs,
            embedding_epochs=args.embed_epochs,
            embedding_window=args.embed_window,
            embedding_lr=args.embed_lr,
            viz_enabled=not args.no_viz
        )
    
    print("\n" + "=" * 80)
    print("NeuroGen 2.0 - Pre-training Configuration")
    print("=" * 80)
    print(f"GPU Device: {config.gpu_device}")
    print(f"Output: {config.output_checkpoint}")
    print(f"\nPhases (only encoder/decoder - SNN learns during main training):")
    print(f"  Decoder: {'Enabled' if config.pretrain_decoder else 'Disabled'}")
    if config.pretrain_decoder:
        print(f"    Iterations: {config.decoder_iterations}")
        print(f"    Base LR: {config.decoder_lr}, Warmup: {config.decoder_lr_warmup}, Decay: {config.decoder_lr_decay}")
    print(f"  Embeddings: {'Enabled' if config.pretrain_embeddings else 'Disabled'}")
    if config.pretrain_embeddings:
        print(f"    Sequences: {config.embedding_sequences}, Epochs: {config.embedding_epochs}")
        print(f"    Window: {config.embedding_window}, LR: {config.embedding_lr}, Neg samples: {config.embedding_negative_samples}")
    print("=" * 80)
    
    try:
        pretrainer = EncoderDecoderPretrainer(config)
        pretrainer.run()
    except KeyboardInterrupt:
        print("\n\nPre-training interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during pre-training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
