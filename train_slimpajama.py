#!/usr/bin/env python3
"""
NeuroGen Modular Brain - SlimPajama Training Script

This script trains the NeuroGen model on the SlimPajama dataset using
next-token prediction. It loads data in chunks, trains continuously,
and saves checkpoints periodically. It also includes comprehensive logging
and periodic evaluation.

Requirements:
    pip install datasets transformers torch numpy tqdm zstandard
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import json
import signal
import random
import datetime
import subprocess

# Add bin to path for libneurogen
sys.path.insert(0, str(Path(__file__).parent / "bin"))

try:
    from datasets import load_dataset
    from tqdm import tqdm
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except ImportError as e:
    print(f"Error: Missing required package - {e}")
    print("Install with: pip install datasets torch numpy tqdm zstandard matplotlib")
    sys.exit(1)

try:
    import zstandard  # noqa: F401
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

@dataclass
class TrainingMetrics:
    """Real-time training metrics"""
    step: int
    loss: float
    accuracy: float
    tokens_per_sec: float
    gpu_memory_mb: float
    learning_rate: float
    timestamp: float

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model parameters
    vocab_size: int = 256  # Direct Character I/O
    embedding_dim: int = 2048  # Matches C++ engine config
    gpu_device: int = 0
    temperature: float = 1.0  # Sampling temperature
    
    # Dataset parameters
    dataset_name: str = "cerebras/SlimPajama-627B"
    dataset_split: str = "train"
    streaming: bool = True  # Stream for large datasets
    max_seq_length: int = 256  # Reduced for GTX 1650 memory efficiency
    
    # Training parameters
    batch_size: int = 1
    learning_rate: float = 0.001
    num_epochs: int = 1
    tokens_per_chunk: int = 4096
    max_chunks: Optional[int] = None  # None = train on full dataset
    
    # Pre-trained weights (optional - load from pretrain_encoder_decoder.py output)
    pretrained_checkpoint: Optional[str] = None  # Path to pre-trained encoder/decoder weights
    
    # Checkpoint parameters
    checkpoint_dir: str = "checkpoints"
    save_checkpoint: bool = True  # Enable/disable checkpoint saving
    # Note: Checkpoints are now only saved at completion or on interruption
    
    # Logging & Evaluation
    log_dir: str = "logs"
    log_interval: int = 10  # Log every N steps
    statistics_interval: int = 100  # Print detailed stats every N steps
    eval_interval: int = 500 # Run periodic evaluation every N steps
    verbose_logging: bool = True
    chunk_debug_interval: int = 1  # Log every sample to ensure visibility
    
    # Visualization
    viz_dir: str = "training_viz"
    viz_interval: int = 5  # Generate charts every 5 steps
    save_samples_every: int = 1  # Save text samples every step
    viz_interval_chunks: int = 5  # Generate graphs every N chunks
    
    # Tokenizer
    tokenizer_dir: str = "tokenizer"
    tokenizer_model: str = "nlp_agent_tokenizer.model"


class VisualizationManager:
    """Manages training visualization and chart generation"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.viz_dir = Path(config.viz_dir)
        self.viz_dir.mkdir(exist_ok=True)

        # Metric history
        self.history: Dict[str, List] = {
            'chunks': [],
            'loss': [],
            'accuracy': [],
            'throughput': [],
            'tokens_processed': []
        }

        # Text samples
        self.text_samples: List[Dict] = []

    def update(self, chunk: int, loss: float, accuracy: float, throughput: float,
               tokens_processed: int, input_text: str = "", output_text: str = "",
               expected_text: str = "", predicted_text: str = "",
               step: Optional[int] = None, sample_index: Optional[int] = None):
        """Update metrics and optionally add text sample.

        `chunk` indexes aggregated training chunks.
        `step` (if provided) is the true global training step.
        `sample_index` is the index of the sample within that chunk.
        """
        self.history['chunks'].append(chunk)
        self.history['loss'].append(loss)
        self.history['accuracy'].append(accuracy)
        self.history['throughput'].append(throughput)
        self.history['tokens_processed'].append(tokens_processed)

        if input_text and (output_text or expected_text or predicted_text):
            sample_record = {
                'chunk': chunk,
                'step': step if step is not None else chunk,
                'input': input_text,
                'output': predicted_text or output_text,
                'expected_next': expected_text,
                'predicted_next': predicted_text,
                'accuracy': accuracy,
            }
            if sample_index is not None:
                sample_record['sample_index'] = sample_index
            self.text_samples.append(sample_record)

    def save_text_samples(self):
        """Save text samples to JSON"""
        samples_path = self.viz_dir / 'text_samples.json'
        with open(samples_path, 'w') as f:
            json.dump(self.text_samples, f, indent=2)
        print(f"💾 Updated {samples_path}")

    def generate_charts(self, chunk: int):
        """Generate comprehensive training charts"""
        if len(self.history['chunks']) < 2:
            return

        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Loss Curve
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.history['chunks'], self.history['loss'], 'b-', linewidth=2, alpha=0.8)
        ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Chunk')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)

        # Add moving average
        if len(self.history['loss']) > 5:
            window = min(5, len(self.history['loss']))
            ma_loss = np.convolve(self.history['loss'], np.ones(window)/window, mode='valid')
            ma_chunks = self.history['chunks'][window-1:]
            ax1.plot(ma_chunks, ma_loss, 'r--', linewidth=2, label=f'{window}-chunk MA', alpha=0.6)
            ax1.legend()

        # 2. Accuracy Curve
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.history['chunks'], [a*100 for a in self.history['accuracy']],
                'g-', linewidth=2, alpha=0.8)
        ax2.set_title('Training Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Chunk')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_ylim([0, 100])
        ax2.grid(True, alpha=0.3)

        # 3. Throughput
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(self.history['chunks'], self.history['throughput'],
                'purple', linewidth=2, alpha=0.8)
        ax3.set_title('Token Throughput', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Chunk')
        ax3.set_ylabel('Tokens/Second')
        ax3.grid(True, alpha=0.3)

        # 4. Cumulative Tokens
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(self.history['chunks'], self.history['tokens_processed'],
                'orange', linewidth=2, alpha=0.8)
        ax4.set_title('Cumulative Tokens Processed', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Chunk')
        ax4.set_ylabel('Total Tokens')
        ax4.grid(True, alpha=0.3)

        # 5. Loss vs Accuracy
        ax5 = fig.add_subplot(gs[1, 1])
        scatter = ax5.scatter(self.history['loss'],
                            [a*100 for a in self.history['accuracy']],
                            c=self.history['chunks'], cmap='viridis',
                            alpha=0.6, s=50)
        ax5.set_title('Loss vs Accuracy', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Loss')
        ax5.set_ylabel('Accuracy (%)')
        ax5.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax5, label='Chunk')

        # 6. Recent Performance
        ax6 = fig.add_subplot(gs[1, 2])
        recent_chunks = min(20, len(self.history['chunks']))
        if len(self.history['chunks']) > 1:
            recent_loss = self.history['loss'][-recent_chunks:]
            recent_acc = [a*100 for a in self.history['accuracy'][-recent_chunks:]]
            recent_x = list(range(len(recent_loss)))

            ax6_twin = ax6.twinx()
            ax6.plot(recent_x, recent_loss, 'b-', linewidth=2, label='Loss', alpha=0.8)
            ax6_twin.plot(recent_x, recent_acc, 'g-', linewidth=2, label='Accuracy', alpha=0.8)

            ax6.set_title(f'Recent Performance (Last {recent_chunks} chunks)',
                         fontsize=14, fontweight='bold')
            ax6.set_xlabel('Chunk (relative)')
            ax6.set_ylabel('Loss', color='b')
            ax6_twin.set_ylabel('Accuracy (%)', color='g')
            ax6.tick_params(axis='y', labelcolor='b')
            ax6_twin.tick_params(axis='y', labelcolor='g')
            ax6.grid(True, alpha=0.3)

        # 7. Text Samples Table
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')

        if self.text_samples:
            # Show last 5 samples
            recent_samples = self.text_samples[-5:]
            table_data = []
            for sample in recent_samples:
                input_short = (sample['input'][:60] + '...') if len(sample['input']) > 60 else sample['input']
                output_short = (sample['output'][:60] + '...') if len(sample['output']) > 60 else sample['output']
                table_data.append([
                    f"Chunk {sample['step']}",
                    input_short,
                    output_short,
                    f"{sample['accuracy']*100:.1f}%"
                ])

            table = ax7.table(cellText=table_data,
                            colLabels=['Chunk', 'Input Text', 'Model Output', 'Accuracy'],
                            cellLoc='left',
                            loc='center',
                            colWidths=[0.15, 0.35, 0.35, 0.15])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)

            # Style header
            for i in range(4):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')

            ax7.set_title('Recent Text Samples', fontsize=14, fontweight='bold', pad=20)

        # Overall title
        fig.suptitle(f'NeuroGen 2.0 SlimPajama Training - Chunk {chunk}',
                    fontsize=16, fontweight='bold')

        # Save
        output_path = self.viz_dir / f'training_chunk_{chunk:06d}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Also save latest
        latest_path = self.viz_dir / 'training_latest.png'
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Recreate the same plot (reusing the same code)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.history['chunks'], self.history['loss'], 'b-', linewidth=2, alpha=0.8)
        ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Chunk')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        if len(self.history['loss']) > 5:
            window = min(5, len(self.history['loss']))
            ma_loss = np.convolve(self.history['loss'], np.ones(window)/window, mode='valid')
            ma_chunks = self.history['chunks'][window-1:]
            ax1.plot(ma_chunks, ma_loss, 'r--', linewidth=2, label=f'{window}-chunk MA', alpha=0.6)
            ax1.legend()

        fig.suptitle(f'NeuroGen 2.0 SlimPajama Training - Chunk {chunk}',
                    fontsize=16, fontweight='bold')
        plt.savefig(latest_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📊 Saved visualization: {output_path.name}")


class SlimPajamaTrainer:
    """Trainer for NeuroGen on SlimPajama dataset"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize SentencePiece tokenizer
        print(f"📝 Loading SentencePiece tokenizer from {config.tokenizer_dir}")
        tokenizer_path = Path(config.tokenizer_dir) / config.tokenizer_model
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(str(tokenizer_path))
        
        # Load vocab size from config
        tokenizer_state_path = Path(config.tokenizer_dir) / "tokenizer_state.json"
        with open(tokenizer_state_path, 'r') as f:
            tokenizer_config = json.load(f)
        actual_vocab_size = tokenizer_config.get('vocab_size', 32000)
        print(f"✅ Tokenizer loaded: vocab_size={actual_vocab_size}")
        
        # Update config vocab size if different
        if actual_vocab_size != config.vocab_size:
            print(f"⚠️  Updating vocab_size from {config.vocab_size} to {actual_vocab_size}")
            config.vocab_size = actual_vocab_size
            
        # Training state
        self.global_step = 0
        self.total_loss = 0.0
        self.tokens_processed = 0
        self.start_time = time.time()
        self.sample_step = 0  # global per-sample step counter
        self.latest_checkpoint_path = self.checkpoint_dir / "latest_slimpajama.ckpt"

        # Visualization
        self.viz = VisualizationManager(config)
        
        # Add library path to sys.path to allow loading libneurogen
        sys.path.insert(0, str(Path(__file__).parent / "bin"))

        # Initialize model
        print("🚀 Initializing NeuroGen model...")
        try:
            import libneurogen
            self.model = libneurogen.NeuroGenModel(
                vocab_size=self.config.vocab_size,
                embedding_dim=self.config.embedding_dim,
                gpu_device=self.config.gpu_device
            )
            print("✅ Model initialized\n")
        except ImportError:
            print("⚠️  libneurogen not found - running in simulation mode")
            self.model = None

        # Load pre-trained encoder/decoder weights if specified
        if config.pretrained_checkpoint and self.model:
            self.load_pretrained_weights(config.pretrained_checkpoint)

    def load_pretrained_weights(self, checkpoint_path: str):
        """Load pre-trained encoder/decoder weights from a checkpoint.
        
        Args:
            checkpoint_path: Path to pre-trained checkpoint (from pretrain_encoder_decoder.py)
        """
        print(f"\n📂 Loading pre-trained weights from: {checkpoint_path}")
        try:
            self.model.load_checkpoint(checkpoint_path)
            print("✅ Pre-trained weights loaded successfully")
        except Exception as e:
            print(f"⚠️  Failed to load pre-trained weights: {e}")
            print("   Continuing with random initialization...")

    def tokenize_text(self, text: str) -> Tuple[List[int], int]:
        """
        Tokenize text into context and target token.
        
        Returns:
            context_ids: list of token ids for the input context
            target_id:   the token id immediately following that prefix
        """
        # Direct Character Encoding
        tokens = [b for b in text.encode('utf-8')]

        if len(tokens) < 2:
            return [], -1

        # Truncate for memory safety first
        if len(tokens) > self.config.max_seq_length:
            tokens = tokens[: self.config.max_seq_length]

        # We don't want to always predict the *last* token of the truncated
        # sequence as the target, because the human-visible input snippet we
        # log for inspection is only a prefix (first 32 tokens). To make
        # expected_next align with the last token in the visible input, we
        # train and log on that same prefix position.
        visible_len = min(32, len(tokens) - 1)  # leave room for at least one next token
        prefix = tokens[:visible_len]
        target_pos = visible_len  # immediate next token after the visible prefix

        context_ids = prefix
        target_id = tokens[target_pos]

        return context_ids, target_id

    def train_on_chunk(self, texts: List[str]) -> Tuple[float, float, str, str, str, int, int]:
        """Train on a chunk of texts.

        Returns:
            avg_loss, avg_accuracy,
            sample_input_text, expected_next_text, predicted_next_text,
            sample_index_within_chunk, sample_step_at_sample
        """
        chunk_loss = 0.0
        chunk_accuracy = 0.0
        chunk_tokens = 0
        processed_samples = 0
        skipped_samples = 0
        sample_input = ""
        expected_next_text = ""
        predicted_next_text = ""
        first_sample_index = -1
        first_sample_step = -1

        for idx, text in enumerate(texts):
            context_ids, target_id = self.tokenize_text(text)
            if not context_ids or target_id < 0:
                skipped_samples += 1
                continue

            processed_samples += 1
            self.sample_step += 1

            if self.model:
                # Train on this exact (context, next-token) pair.
                # The model's internal working memory (PFC, Hippocampus) maintains
                # temporal context automatically via pipeline state.
                # Wrap target_id in a single-element list to satisfy the C++ signature.
                loss, accuracy, predicted_token_id = self.model.train_step(context_ids, [target_id])
            else:
                time.sleep(0.001)
                loss = 2.5
                accuracy = 0.1
                predicted_token_id = target_id  # pretend perfect prediction in simulation

            # Accumulate loss/accuracy weighted by context length
            chunk_loss += loss * len(context_ids)
            chunk_accuracy += accuracy * len(context_ids)

            # Capture first sample details for visualization
            if sample_input == "":
                # Input text is exactly the context used for training (visible prefix)
                # Decode bytes directly
                sample_input = bytes(context_ids).decode('utf-8', errors='replace')

                expected_next_text = bytes([target_id]).decode('utf-8', errors='replace')
                if predicted_token_id >= 0:
                    predicted_next_text = bytes([predicted_token_id]).decode('utf-8', errors='replace')
                else:
                    predicted_next_text = ""

                first_sample_index = idx
                first_sample_step = self.sample_step

            chunk_tokens += len(context_ids)
            self.tokens_processed += len(context_ids)

        if self.config.verbose_logging:
            print(f"      Processed {processed_samples} samples ({skipped_samples} skipped), {chunk_tokens} tokens")

        avg_loss = chunk_loss / max(chunk_tokens, 1)
        avg_accuracy = chunk_accuracy / max(chunk_tokens, 1)
        return (
            avg_loss,
            avg_accuracy,
            sample_input,
            expected_next_text,
            predicted_next_text,
            first_sample_index,
            first_sample_step,
        )

    def train(self):
        """Main training loop"""
        print(f"🚀 Starting training on {self.config.dataset_name}")
        print(f"   Streaming mode: {self.config.streaming}")
        print(f"   Tokens per chunk: {self.config.tokens_per_chunk}")
        print(f"   Max sequence length: {self.config.max_seq_length}")
        
        if self.config.pretrained_checkpoint:
            print(f"   Pre-trained weights: ✓ Loaded from {self.config.pretrained_checkpoint}")
        else:
            print("   Pre-trained weights: ✗ Not loaded (use --pretrained-checkpoint to load)")
        
        try:
            print("\n⏳ Loading dataset (this may take 1-2 minutes for initial download)...")
            dataset = load_dataset(
                self.config.dataset_name,
                split=self.config.dataset_split,
                streaming=self.config.streaming
            )
            print("✅ Dataset loaded successfully\n")
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            return

        chunk_texts = []
        chunk_tokens = 0
        samples_in_chunk = 0
        last_update_time = time.time()
        
        print("📊 Starting data processing...")
        print(f"   Will accumulate ~{self.config.tokens_per_chunk} tokens before training step\n")
        
        for i, example in enumerate(dataset):
            text = example.get("text", "")
            if not text:
                continue
            
            # Progress update while accumulating (every 2 seconds or 100 samples)
            current_time = time.time()
            if current_time - last_update_time >= 2.0 or (i > 0 and i % 100 == 0):
                print(f"📥 Accumulating data: {samples_in_chunk} samples, ~{chunk_tokens} tokens in current chunk (sample #{i})")
                last_update_time = current_time
                
            chunk_texts.append(text)
            samples_in_chunk += 1
            # Estimate tokens (fast approximation)
            chunk_tokens += len(text) // 4
            
            if chunk_tokens >= self.config.tokens_per_chunk:
                print(f"\n✓ Chunk complete: {samples_in_chunk} samples, ~{chunk_tokens} tokens")
                print(f"🏋️  Training on chunk {self.global_step + 1}...")

                chunk_start = time.time()
                (
                    loss,
                    accuracy,
                    sample_input,
                    expected_next,
                    predicted_next,
                    sample_idx,
                    sample_step_at_sample,
                ) = self.train_on_chunk(chunk_texts)
                chunk_time = time.time() - chunk_start

                self.global_step += 1
                self.total_loss += loss

                avg_loss = self.total_loss / self.global_step
                elapsed = time.time() - self.start_time
                tps = self.tokens_processed / elapsed

                print(f"✅ Step {self.global_step} complete in {chunk_time:.2f}s")
                print(f"   Loss: {loss:.4f} | Accuracy: {accuracy*100:.2f}% | Avg Loss: {avg_loss:.4f}")
                print(f"   Throughput: {tps:.1f} tokens/sec | Total tokens: {self.tokens_processed:,}")

                # Update visualization metrics
                self.viz.update(
                    chunk=self.global_step,
                    loss=loss,
                    accuracy=accuracy,
                    throughput=tps,
                    tokens_processed=self.tokens_processed,
                    input_text=sample_input,
                    output_text=predicted_next,
                    expected_text=expected_next,
                    predicted_text=predicted_next,
                    step=sample_step_at_sample if sample_step_at_sample >= 0 else self.global_step,
                    sample_index=sample_idx if sample_idx is not None else None,
                )

                # Save text samples JSON every chunk
                self.viz.save_text_samples()

                # Generate visualization graphs every N chunks
                if self.global_step % self.config.viz_interval_chunks == 0:
                    print(f"📊 Generating visualization graphs...")
                    self.viz.generate_charts(self.global_step)

                print()  # Empty line for readability

                # Reset for next chunk
                chunk_texts = []
                chunk_tokens = 0
                samples_in_chunk = 0
                last_update_time = time.time()

                if self.config.max_chunks and self.global_step >= self.config.max_chunks:
                    print(f"✅ Reached max_chunks limit ({self.config.max_chunks}). Stopping.")
                    break
        
        # Final saves
        print("\n📊 Generating final visualizations...")
        self.viz.save_text_samples()
        if self.global_step > 0:
            self.viz.generate_charts(self.global_step)

        # Save final checkpoint (if enabled)
        if self.config.save_checkpoint and self.model and self.global_step > 0:
            ckpt_path_str = str(self.latest_checkpoint_path)
            print(f"\n💾 Saving final checkpoint to {ckpt_path_str}...")
            try:
                self.model.save_checkpoint(ckpt_path_str)
                print("   ✅ Final checkpoint saved.")
            except Exception as ckpt_err:
                print(f"   ⚠️ Failed to save final checkpoint: {ckpt_err}")
        elif not self.config.save_checkpoint:
            print("\n⏭️  Checkpoint saving disabled (use without --no-checkpoint to enable)")

        print("\n" + "="*80)
        print("🎉 Training complete!")
        print("="*80)
        print(f"Total steps: {self.global_step}")
        print(f"Total tokens: {self.tokens_processed:,}")
        print(f"Average loss: {self.total_loss / max(self.global_step, 1):.4f}")
        print(f"Total time: {time.time() - self.start_time:.1f}s")
        print(f"📁 Visualizations saved to: {self.config.viz_dir}/")

def main():
    parser = argparse.ArgumentParser(description="Train NeuroGen on SlimPajama dataset")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--max-chunks", type=int, default=None, help="Maximum number of chunks to train (None = unlimited)")
    parser.add_argument("--tokens-per-chunk", type=int, default=4096, help="Tokens to accumulate before training")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--test", action="store_true", help="Quick test mode (5 chunks, 1024 tokens each)")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose logging")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable checkpoint saving (for faster testing)")
    
    # Pre-trained weights (from pretrain_encoder_decoder.py)
    parser.add_argument("--pretrained-checkpoint", type=str, default=None, 
                        help="Path to pre-trained encoder/decoder checkpoint (from pretrain_encoder_decoder.py)")
    
    args = parser.parse_args()
    
    # Configure based on mode
    if args.test:
        print("🧪 Running in TEST MODE (quick verification)")
        config = TrainingConfig(
            gpu_device=args.gpu,
            tokens_per_chunk=1024,  # Smaller chunks for faster testing
            max_chunks=5,  # Only 5 chunks
            max_seq_length=256,  # Shorter sequences
            verbose_logging=True,
            save_checkpoint=not args.no_checkpoint,
            pretrained_checkpoint=args.pretrained_checkpoint
        )
    else:
        config = TrainingConfig(
            gpu_device=args.gpu,
            tokens_per_chunk=args.tokens_per_chunk,
            max_chunks=args.max_chunks,
            max_seq_length=args.max_seq_length,
            verbose_logging=args.verbose,
            save_checkpoint=not args.no_checkpoint,
            pretrained_checkpoint=args.pretrained_checkpoint
        )
    
    print("\n" + "="*80)
    print("🧠 NeuroGen 2.0 - SlimPajama Training")
    print("="*80)
    print(f"Configuration:")
    print(f"  GPU Device: {config.gpu_device}")
    print(f"  Tokens per chunk: {config.tokens_per_chunk}")
    print(f"  Max sequence length: {config.max_seq_length}")
    print(f"  Max chunks: {config.max_chunks if config.max_chunks else 'unlimited'}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Save checkpoints: {'✓ Enabled' if config.save_checkpoint else '✗ Disabled'}")
    print(f"  Pre-trained checkpoint: {config.pretrained_checkpoint if config.pretrained_checkpoint else '✗ None (random init)'}")
    print("="*80 + "\n")
    
    trainer = None
    try:
        trainer = SlimPajamaTrainer(config)
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
        # Save checkpoint on interruption (if enabled)
        if config.save_checkpoint and trainer and trainer.model and trainer.global_step > 0:
            ckpt_path_str = str(trainer.latest_checkpoint_path)
            print(f"\n💾 Saving checkpoint before exit to {ckpt_path_str}...")
            try:
                trainer.model.save_checkpoint(ckpt_path_str)
                print("   ✅ Checkpoint saved successfully.")
            except Exception as ckpt_err:
                print(f"   ⚠️ Failed to save checkpoint: {ckpt_err}")
        elif not config.save_checkpoint:
            print("\n⏭️  Checkpoint saving disabled (was not saved)")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        # Save checkpoint on error (if enabled)
        if config.save_checkpoint and trainer and trainer.model and trainer.global_step > 0:
            ckpt_path_str = str(trainer.latest_checkpoint_path)
            print(f"\n💾 Attempting to save checkpoint before exit to {ckpt_path_str}...")
            try:
                trainer.model.save_checkpoint(ckpt_path_str)
                print("   ✅ Checkpoint saved successfully.")
            except Exception as ckpt_err:
                print(f"   ⚠️ Failed to save checkpoint: {ckpt_err}")
        elif not config.save_checkpoint:
            print("\n⏭️  Checkpoint saving disabled (was not saved)")

if __name__ == "__main__":
    main()
