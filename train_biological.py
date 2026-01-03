#!/usr/bin/env python3
"""
NeuroGen 2.5 - Biological Curriculum Training Script

This script implements a comprehensive curriculum learning approach for training
the NeuroGen spiking neural network on natural language. The curriculum progresses
through three distinct phases:

Phase 1: CHARACTER ASSOCIATIONS (Foundation)
    - Learn individual character patterns
    - Build basic character-to-character transitions
    - Establish fundamental neural representations

Phase 2: WORD FORMATION (Structure)  
    - Learn common word patterns and boundaries
    - Build word-level representations
    - Understand word structure and morphology

Phase 3: SENTENCE UNDERSTANDING (Semantics)
    - Learn sentence structure and grammar
    - Build contextual understanding
    - Generate coherent text sequences

The training uses:
- Hybrid backpropagation: SNN STDP + Decoder gradient descent
- Curriculum learning with automatic phase transitions
- Comprehensive logging and visualization
- SlimPajama dataset for real-world text

Usage:
    python train_biological.py --phase 1 --max-epochs 10
    python train_biological.py --phase all --max-hours 24
    python train_biological.py --resume checkpoints/bio_phase2_epoch5.json

Author: NeuroGen Team
Date: December 2025
"""

import os
import sys
import time
import json
import argparse
import random
import datetime
import logging
import traceback
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Generator
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from collections import defaultdict
import numpy as np

# Add bin to path for libneurogen
sys.path.insert(0, str(Path(__file__).parent / "bin"))

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch
    import matplotlib.dates as mdates
except ImportError:
    print("Warning: matplotlib not available, visualization disabled")
    plt = None

try:
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: Missing required package - {e}")
    print("Install with: pip install datasets tqdm matplotlib numpy")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainingPhase(Enum):
    """Curriculum learning phases"""
    CHARACTER = 1  # Character-level associations
    WORD = 2       # Word-level patterns
    SENTENCE = 3   # Sentence-level understanding


@dataclass
class PhaseConfig:
    """Configuration for a training phase"""
    name: str
    description: str
    min_context_len: int
    max_context_len: int
    target_accuracy: float  # Accuracy threshold to advance
    min_epochs: int         # Minimum epochs before advancing
    max_epochs: int         # Maximum epochs in this phase
    learning_rate: float
    reward_scale: float     # Scale for STDP reward signal
    batch_size: int
    samples_per_epoch: int
    
    # Data filtering
    use_simple_text: bool = False  # Use simplified text for early phases
    max_word_length: int = 0       # 0 = no limit
    vocabulary_size: int = 0       # 0 = full vocabulary


@dataclass
class TrainingConfig:
    """Master training configuration"""
    # GPU
    gpu_device: int = 0
    
    # Dataset
    dataset_name: str = "cerebras/SlimPajama-627B"
    dataset_split: str = "train"
    streaming: bool = True
    
    # Curriculum phases
    phases: Dict[TrainingPhase, PhaseConfig] = field(default_factory=dict)
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval_epochs: int = 1
    checkpoint_interval_minutes: int = 30
    
    # Logging
    log_dir: str = "logs/biological"
    log_level: str = "INFO"
    
    # Visualization
    viz_dir: str = "training_viz/biological"
    viz_interval_steps: int = 100
    
    # Training limits
    max_total_hours: Optional[float] = None
    max_total_tokens: Optional[int] = None
    
    # Early stopping
    patience_epochs: int = 5  # Epochs without improvement before stopping
    
    def __post_init__(self):
        """Initialize default phase configurations"""
        if not self.phases:
            self.phases = {
                TrainingPhase.CHARACTER: PhaseConfig(
                    name="Character Associations",
                    description="Learning individual characters and basic transitions",
                    min_context_len=1,
                    max_context_len=16,
                    target_accuracy=0.40,  # 40% accuracy to advance (more realistic)
                    min_epochs=10,
                    max_epochs=100,
                    learning_rate=0.01,
                    reward_scale=0.5,
                    batch_size=64,
                    samples_per_epoch=200000,  # More samples per epoch
                    use_simple_text=True,
                    max_word_length=8,
                    vocabulary_size=128,  # ASCII printable
                ),
                TrainingPhase.WORD: PhaseConfig(
                    name="Word Formation",
                    description="Learning word patterns and boundaries",
                    min_context_len=4,
                    max_context_len=64,
                    target_accuracy=0.30,  # 30% accuracy to advance (realistic for words)
                    min_epochs=20,
                    max_epochs=200,
                    learning_rate=0.005,
                    reward_scale=0.3,
                    batch_size=32,
                    samples_per_epoch=500000,
                    use_simple_text=False,
                    max_word_length=0,
                    vocabulary_size=256,
                ),
                TrainingPhase.SENTENCE: PhaseConfig(
                    name="Sentence Understanding",
                    description="Learning sentence structure and generation",
                    min_context_len=16,
                    max_context_len=256,
                    target_accuracy=0.25,  # Lower target for complex task
                    min_epochs=30,
                    max_epochs=500,
                    learning_rate=0.001,
                    reward_scale=0.2,
                    batch_size=16,
                    samples_per_epoch=1000000,
                    use_simple_text=False,
                    max_word_length=0,
                    vocabulary_size=256,
                ),
            }


# ============================================================================
# METRICS AND LOGGING
# ============================================================================

@dataclass
class EpochMetrics:
    """Metrics for a single epoch"""
    epoch: int
    phase: TrainingPhase
    timestamp: str
    duration_seconds: float
    
    # Loss metrics
    avg_loss: float
    min_loss: float
    max_loss: float
    
    # Accuracy metrics
    avg_accuracy: float
    char_accuracy: float  # Per-character accuracy
    word_accuracy: float  # Per-word accuracy (if applicable)
    
    # Throughput
    tokens_processed: int
    tokens_per_second: float
    samples_processed: int
    
    # Neural activity
    avg_sparsity: float
    avg_entropy: float
    
    # Learning dynamics
    reward_given: float
    weight_change_magnitude: float


@dataclass  
class PhaseMetrics:
    """Aggregated metrics for a training phase"""
    phase: TrainingPhase
    start_time: str
    end_time: Optional[str] = None
    
    epochs_completed: int = 0
    total_tokens: int = 0
    total_samples: int = 0
    
    best_accuracy: float = 0.0
    best_loss: float = float('inf')
    final_accuracy: float = 0.0
    
    epoch_history: List[EpochMetrics] = field(default_factory=list)
    
    # Convergence tracking
    accuracy_history: List[float] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)
    
    def is_converged(self, config: PhaseConfig) -> Tuple[bool, str]:
        """Check if phase has converged (ready to advance)"""
        if self.epochs_completed < config.min_epochs:
            return False, f"Need {config.min_epochs - self.epochs_completed} more epochs"
        
        if self.best_accuracy >= config.target_accuracy:
            return True, f"Target accuracy {config.target_accuracy:.1%} achieved"
        
        if self.epochs_completed >= config.max_epochs:
            return True, f"Max epochs ({config.max_epochs}) reached"
        
        return False, "Training..."


@dataclass
class TrainingState:
    """Complete training state for checkpointing"""
    config: Dict[str, Any]
    current_phase: int  # TrainingPhase value
    current_epoch: int
    global_step: int
    total_tokens: int
    total_time_seconds: float
    
    phase_metrics: Dict[int, Dict] = field(default_factory=dict)
    
    # Recent samples for visualization
    recent_samples: List[Dict] = field(default_factory=list)
    
    # Best model tracking
    best_accuracy: float = 0.0
    best_loss: float = float('inf')
    best_checkpoint_path: Optional[str] = None


class TrainingLogger:
    """Comprehensive logging for biological training"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"training_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("BiologicalTraining")
        
        # Metrics file
        self.metrics_file = self.log_dir / f"metrics_{timestamp}.jsonl"
        
        # Summary file
        self.summary_file = self.log_dir / f"summary_{timestamp}.json"
        
    def log_epoch(self, metrics: EpochMetrics):
        """Log epoch metrics"""
        self.logger.info(
            f"Epoch {metrics.epoch} | Phase {metrics.phase.name} | "
            f"Loss: {metrics.avg_loss:.4f} | Acc: {metrics.avg_accuracy:.2%} | "
            f"Tokens: {metrics.tokens_processed:,} @ {metrics.tokens_per_second:.1f}/s"
        )
        
        # Append to metrics file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(asdict(metrics), default=str) + '\n')
    
    def log_phase_start(self, phase: TrainingPhase, config: PhaseConfig):
        """Log phase start"""
        self.logger.info("=" * 80)
        self.logger.info(f"🚀 Starting Phase {phase.value}: {config.name}")
        self.logger.info(f"   {config.description}")
        self.logger.info(f"   Context: {config.min_context_len}-{config.max_context_len} chars")
        self.logger.info(f"   Target accuracy: {config.target_accuracy:.1%}")
        self.logger.info(f"   Learning rate: {config.learning_rate}")
        self.logger.info("=" * 80)
    
    def log_phase_end(self, phase: TrainingPhase, metrics: PhaseMetrics, reason: str):
        """Log phase completion"""
        self.logger.info("=" * 80)
        self.logger.info(f"✅ Completed Phase {phase.value}: {phase.name}")
        self.logger.info(f"   Reason: {reason}")
        self.logger.info(f"   Epochs: {metrics.epochs_completed}")
        self.logger.info(f"   Best accuracy: {metrics.best_accuracy:.2%}")
        self.logger.info(f"   Total tokens: {metrics.total_tokens:,}")
        self.logger.info("=" * 80)
    
    def log_sample(self, input_text: str, expected: str, predicted: str, correct: bool):
        """Log a training sample"""
        status = "✓" if correct else "✗"
        # Escape special characters for logging
        input_repr = repr(input_text)[1:-1][:50]
        expected_repr = repr(expected)[1:-1]
        predicted_repr = repr(predicted)[1:-1]
        self.logger.debug(
            f"{status} Input: '{input_repr}' → Expected: '{expected_repr}' | Predicted: '{predicted_repr}'"
        )
    
    def save_summary(self, state: TrainingState):
        """Save training summary"""
        summary = {
            'completed_at': datetime.datetime.now().isoformat(),
            'current_phase': state.current_phase,
            'total_epochs': state.current_epoch,
            'total_tokens': state.total_tokens,
            'total_time_hours': state.total_time_seconds / 3600,
            'best_accuracy': state.best_accuracy,
            'best_loss': state.best_loss,
            'phase_summaries': {}
        }
        
        for phase_val, metrics_dict in state.phase_metrics.items():
            summary['phase_summaries'][phase_val] = {
                'epochs': metrics_dict.get('epochs_completed', 0),
                'best_accuracy': metrics_dict.get('best_accuracy', 0),
                'total_tokens': metrics_dict.get('total_tokens', 0),
            }
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)


# ============================================================================
# VISUALIZATION
# ============================================================================

class BiologicalVisualizer:
    """Advanced visualization for biological training"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.viz_dir = Path(config.viz_dir)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Color scheme for phases
        self.phase_colors = {
            TrainingPhase.CHARACTER: '#FF6B6B',  # Red
            TrainingPhase.WORD: '#4ECDC4',       # Teal
            TrainingPhase.SENTENCE: '#45B7D1',   # Blue
        }
        
        # History for plotting
        self.history = {
            'steps': [],
            'loss': [],
            'accuracy': [],
            'phase': [],
            'entropy': [],
            'sparsity': [],
            'tokens_per_sec': [],
        }
        
        # Text samples
        self.samples = []
        
    def update(self, step: int, loss: float, accuracy: float, phase: TrainingPhase,
               entropy: float = 0.0, sparsity: float = 0.0, tps: float = 0.0,
               sample: Optional[Dict] = None):
        """Update visualization history"""
        self.history['steps'].append(step)
        self.history['loss'].append(loss)
        self.history['accuracy'].append(accuracy)
        self.history['phase'].append(phase.value)
        self.history['entropy'].append(entropy)
        self.history['sparsity'].append(sparsity)
        self.history['tokens_per_sec'].append(tps)
        
        if sample:
            self.samples.append(sample)
            # Keep only recent samples
            if len(self.samples) > 1000:
                self.samples = self.samples[-500:]
    
    def generate_dashboard(self, state: TrainingState, epoch_metrics: Optional[EpochMetrics] = None):
        """Generate comprehensive training dashboard"""
        if plt is None:
            return
        
        if len(self.history['steps']) < 2:
            return
        
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)
        
        # Title
        phase_name = TrainingPhase(state.current_phase).name
        fig.suptitle(
            f'NeuroGen 2.5 Biological Training Dashboard\n'
            f'Phase {state.current_phase}: {phase_name} | Epoch {state.current_epoch} | '
            f'Step {state.global_step:,}',
            fontsize=16, fontweight='bold'
        )
        
        # 1. Loss over time (colored by phase)
        ax1 = fig.add_subplot(gs[0, 0:2])
        self._plot_metric_by_phase(ax1, 'loss', 'Training Loss', 'Loss')
        
        # 2. Accuracy over time (colored by phase)
        ax2 = fig.add_subplot(gs[0, 2:4])
        self._plot_metric_by_phase(ax2, 'accuracy', 'Training Accuracy', 'Accuracy', pct=True)
        
        # 3. Phase progress
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_phase_progress(ax3, state)
        
        # 4. Neural activity (entropy/sparsity)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_neural_activity(ax4)
        
        # 5. Throughput
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_throughput(ax5)
        
        # 6. Learning curves per phase
        ax6 = fig.add_subplot(gs[1, 3])
        self._plot_per_phase_curves(ax6, state)
        
        # 7. Recent accuracy distribution
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_accuracy_distribution(ax7)
        
        # 8. Character-level analysis
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_character_analysis(ax8)
        
        # 9. Training statistics
        ax9 = fig.add_subplot(gs[2, 2:4])
        self._plot_statistics_table(ax9, state, epoch_metrics)
        
        # 10. Sample predictions table
        ax10 = fig.add_subplot(gs[3, :])
        self._plot_samples_table(ax10)
        
        # Save
        output_path = self.viz_dir / f'dashboard_step_{state.global_step:08d}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Also save as latest
        latest_path = self.viz_dir / 'dashboard_latest.png'
        plt.savefig(latest_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    def _plot_metric_by_phase(self, ax, metric: str, title: str, ylabel: str, pct: bool = False):
        """Plot a metric with phase coloring"""
        steps = self.history['steps']
        values = self.history[metric]
        phases = self.history['phase']
        
        if pct:
            values = [v * 100 for v in values]
        
        # Plot each phase segment
        for phase in TrainingPhase:
            mask = [p == phase.value for p in phases]
            phase_steps = [s for s, m in zip(steps, mask) if m]
            phase_values = [v for v, m in zip(values, mask) if m]
            if phase_steps:
                ax.scatter(phase_steps, phase_values, 
                          c=self.phase_colors[phase], alpha=0.3, s=10, label=phase.name)
        
        # Add smoothed line
        if len(values) > 10:
            window = min(50, len(values) // 5)
            smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
            smooth_steps = steps[window-1:]
            ax.plot(smooth_steps, smoothed, 'k-', linewidth=2, alpha=0.8, label='Smoothed')
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel(ylabel + (' (%)' if pct else ''))
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_phase_progress(self, ax, state: TrainingState):
        """Plot phase progress bars"""
        phases = list(TrainingPhase)
        current = state.current_phase
        
        for i, phase in enumerate(phases):
            color = self.phase_colors[phase]
            if phase.value < current:
                # Completed
                ax.barh(i, 1.0, color=color, alpha=0.8)
                ax.text(0.5, i, '✓ Complete', ha='center', va='center', 
                       fontweight='bold', color='white')
            elif phase.value == current:
                # Current
                # Get progress from metrics
                progress = 0.5  # Default
                if phase.value in state.phase_metrics:
                    pm = state.phase_metrics[phase.value]
                    epochs = pm.get('epochs_completed', 0)
                    max_epochs = self.config.phases[phase].max_epochs
                    progress = min(1.0, epochs / max_epochs)
                
                ax.barh(i, progress, color=color, alpha=0.8)
                ax.barh(i, 1.0, color=color, alpha=0.2)
                ax.text(0.5, i, f'{progress:.0%}', ha='center', va='center',
                       fontweight='bold')
            else:
                # Future
                ax.barh(i, 1.0, color='lightgray', alpha=0.3)
                ax.text(0.5, i, 'Pending', ha='center', va='center',
                       color='gray')
        
        ax.set_yticks(range(len(phases)))
        ax.set_yticklabels([p.name for p in phases])
        ax.set_xlim(0, 1)
        ax.set_title('Phase Progress', fontweight='bold')
        ax.set_xlabel('Completion')
    
    def _plot_neural_activity(self, ax):
        """Plot neural activity metrics"""
        if not self.history['entropy']:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            return
        
        steps = self.history['steps'][-200:]
        entropy = self.history['entropy'][-200:]
        sparsity = self.history['sparsity'][-200:]
        
        ax2 = ax.twinx()
        
        l1, = ax.plot(steps, entropy, 'b-', alpha=0.7, label='Entropy')
        l2, = ax2.plot(steps, sparsity, 'r-', alpha=0.7, label='Sparsity')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Entropy (bits)', color='b')
        ax2.set_ylabel('Sparsity', color='r')
        ax.set_title('Neural Activity', fontweight='bold')
        ax.legend(handles=[l1, l2], loc='upper right', fontsize=8)
    
    def _plot_throughput(self, ax):
        """Plot throughput over time"""
        if not self.history['tokens_per_sec']:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            return
        
        steps = self.history['steps'][-100:]
        tps = self.history['tokens_per_sec'][-100:]
        
        ax.fill_between(steps, tps, alpha=0.3, color='green')
        ax.plot(steps, tps, 'g-', linewidth=2)
        
        avg_tps = np.mean(tps) if tps else 0
        ax.axhline(avg_tps, color='darkgreen', linestyle='--', label=f'Avg: {avg_tps:.1f}')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Tokens/Second')
        ax.set_title('Throughput', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_per_phase_curves(self, ax, state: TrainingState):
        """Plot learning curves per phase"""
        for phase in TrainingPhase:
            if phase.value in state.phase_metrics:
                pm = state.phase_metrics[phase.value]
                acc_history = pm.get('accuracy_history', [])
                if acc_history:
                    ax.plot(range(len(acc_history)), [a*100 for a in acc_history],
                           color=self.phase_colors[phase], linewidth=2, 
                           label=f'{phase.name}', marker='o', markersize=4)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Per-Phase Learning Curves', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_accuracy_distribution(self, ax):
        """Plot distribution of recent accuracies"""
        recent_acc = self.history['accuracy'][-500:]
        if not recent_acc:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            return
        
        ax.hist([a*100 for a in recent_acc], bins=20, color='steelblue', 
                alpha=0.7, edgecolor='white')
        
        mean_acc = np.mean(recent_acc) * 100
        ax.axvline(mean_acc, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_acc:.1f}%')
        
        ax.set_xlabel('Accuracy (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Recent Accuracy Distribution', fontweight='bold')
        ax.legend(fontsize=8)
    
    def _plot_character_analysis(self, ax):
        """Analyze per-character accuracy"""
        if not self.samples:
            ax.text(0.5, 0.5, 'No samples', ha='center', va='center')
            ax.set_title('Character Analysis', fontweight='bold')
            return
        
        # Count correct/incorrect per character
        char_correct = defaultdict(int)
        char_total = defaultdict(int)
        
        for sample in self.samples[-200:]:
            expected = sample.get('expected', '')
            predicted = sample.get('predicted', '')
            if expected:
                char_total[expected] += 1
                if expected == predicted:
                    char_correct[expected] += 1
        
        # Get most common characters
        top_chars = sorted(char_total.keys(), key=lambda c: char_total[c], reverse=True)[:15]
        
        if not top_chars:
            ax.text(0.5, 0.5, 'No character data', ha='center', va='center')
            return
        
        accuracies = [char_correct[c] / char_total[c] * 100 if char_total[c] > 0 else 0 
                     for c in top_chars]
        
        # Display chars safely
        display_chars = []
        for c in top_chars:
            if c == ' ':
                display_chars.append('␣')
            elif c == '\n':
                display_chars.append('↵')
            elif ord(c) < 32:
                display_chars.append(f'\\x{ord(c):02x}')
            else:
                display_chars.append(c)
        
        bars = ax.bar(range(len(top_chars)), accuracies, color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(top_chars)))
        ax.set_xticklabels(display_chars, fontsize=10)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Per-Character Accuracy (Top 15)', fontweight='bold')
        ax.set_ylim(0, 100)
        
        # Color bars by accuracy
        for bar, acc in zip(bars, accuracies):
            if acc >= 70:
                bar.set_color('green')
            elif acc >= 40:
                bar.set_color('orange')
            else:
                bar.set_color('red')
    
    def _plot_statistics_table(self, ax, state: TrainingState, epoch_metrics: Optional[EpochMetrics]):
        """Plot statistics table"""
        ax.axis('off')
        
        # Compute statistics
        total_time_hrs = state.total_time_seconds / 3600
        avg_tps = np.mean(self.history['tokens_per_sec'][-100:]) if self.history['tokens_per_sec'] else 0
        recent_acc = np.mean(self.history['accuracy'][-100:]) * 100 if self.history['accuracy'] else 0
        recent_loss = np.mean(self.history['loss'][-100:]) if self.history['loss'] else 0
        
        stats = [
            ['Metric', 'Value'],
            ['Current Phase', TrainingPhase(state.current_phase).name],
            ['Current Epoch', str(state.current_epoch)],
            ['Global Step', f'{state.global_step:,}'],
            ['Total Tokens', f'{state.total_tokens:,}'],
            ['Total Time', f'{total_time_hrs:.2f} hours'],
            ['Avg Throughput', f'{avg_tps:.1f} tok/s'],
            ['Recent Accuracy', f'{recent_acc:.2f}%'],
            ['Recent Loss', f'{recent_loss:.4f}'],
            ['Best Accuracy', f'{state.best_accuracy*100:.2f}%'],
            ['Best Loss', f'{state.best_loss:.4f}'],
        ]
        
        table = ax.table(cellText=stats[1:], colLabels=stats[0],
                        loc='center', cellLoc='left',
                        colWidths=[0.4, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Training Statistics', fontweight='bold', pad=20)
    
    def _plot_samples_table(self, ax):
        """Plot recent samples table"""
        ax.axis('off')
        
        if not self.samples:
            ax.text(0.5, 0.5, 'No samples yet', ha='center', va='center', fontsize=12)
            return
        
        # Get last 8 samples
        recent = self.samples[-8:]
        
        def safe_repr(s, max_len=40):
            """Safe string representation for display"""
            if not s:
                return '""'
            # Handle special characters explicitly
            result = []
            for c in s:
                if c == ' ':
                    result.append('␣')
                elif c == '\n':
                    result.append('↵')
                elif c == '\t':
                    result.append('→')
                elif c == '\r':
                    result.append('⏎')
                elif ord(c) < 32:
                    result.append(f'[{ord(c):02x}]')
                elif ord(c) > 126 and ord(c) < 256:
                    result.append(f'[{ord(c):02x}]')
                else:
                    result.append(c)
            r = ''.join(result)
            if len(r) > max_len:
                r = r[:max_len-3] + '...'
            return r
        
        def safe_char(s):
            """Safe single character representation"""
            if not s:
                return '?'
            c = s[0] if len(s) >= 1 else '?'
            if c == ' ':
                return '␣'
            elif c == '\n':
                return '↵'
            elif c == '\t':
                return '→'
            elif c == '\r':
                return '⏎'
            elif c.startswith('\\'):
                return c[:6]  # Already escaped
            elif ord(c) < 32:
                return f'[{ord(c):02x}]'
            elif ord(c) > 126:
                return f'[{ord(c):02x}]'
            return c
        
        table_data = []
        for sample in recent:
            input_text = safe_repr(sample.get('input', ''), 35)
            expected = safe_char(sample.get('expected', ''))
            predicted = safe_char(sample.get('predicted', ''))
            correct = '✓' if sample.get('correct', False) else '✗'
            
            table_data.append([input_text, expected, predicted, correct])
        
        table = ax.table(
            cellText=table_data,
            colLabels=['Input Context', 'Expected', 'Predicted', ''],
            loc='center',
            cellLoc='left',
            colWidths=[0.55, 0.15, 0.15, 0.05]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color correct/incorrect
        for row_idx, sample in enumerate(recent, 1):
            if sample.get('correct', False):
                table[(row_idx, 3)].set_facecolor('#90EE90')
            else:
                table[(row_idx, 3)].set_facecolor('#FFB6C1')
        
        ax.set_title('Recent Predictions', fontweight='bold', pad=20)
    
    def save_samples_json(self):
        """Save samples to JSON"""
        samples_path = self.viz_dir / 'samples.json'
        with open(samples_path, 'w') as f:
            json.dump(self.samples[-1000:], f, indent=2)


# ============================================================================
# DATA PROCESSING
# ============================================================================

class CurriculumDataLoader:
    """Data loader with curriculum-aware filtering"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.dataset = None
        self.dataset_iter = None
        
        # Simple text patterns for Phase 1
        self.simple_patterns = [
            "the ", "and ", "is ", "to ", "of ", "in ", "a ", "that ",
            "it ", "was ", "for ", "on ", "are ", "as ", "with ", "his ",
            "they ", "be ", "at ", "one ", "have ", "this ", "from ",
        ]
        
        # Common words for filtering
        self.common_words = set([
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
        ])
    
    def initialize(self):
        """Initialize dataset"""
        print("⏳ Loading SlimPajama dataset (streaming mode)...")
        self.dataset = load_dataset(
            self.config.dataset_name,
            split=self.config.dataset_split,
            streaming=self.config.streaming
        )
        self.dataset_iter = iter(self.dataset)
        print("✅ Dataset loaded")
    
    def get_samples(self, phase: TrainingPhase, num_samples: int) -> Generator[Tuple[str, str], None, None]:
        """Generate samples appropriate for the current phase"""
        phase_config = self.config.phases[phase]
        samples_yielded = 0
        
        while samples_yielded < num_samples:
            try:
                example = next(self.dataset_iter)
            except StopIteration:
                # Restart dataset
                self.dataset_iter = iter(self.dataset)
                example = next(self.dataset_iter)
            
            text = example.get("text", "")
            if not text or len(text) < phase_config.min_context_len + 1:
                continue
            
            # Filter based on phase
            if phase == TrainingPhase.CHARACTER:
                # Use simple, repetitive text for character learning
                samples = self._extract_character_samples(text, phase_config)
            elif phase == TrainingPhase.WORD:
                samples = self._extract_word_samples(text, phase_config)
            else:
                samples = self._extract_sentence_samples(text, phase_config)
            
            for context, target in samples:
                if samples_yielded >= num_samples:
                    break
                yield context, target
                samples_yielded += 1
    
    def _extract_character_samples(self, text: str, config: PhaseConfig) -> List[Tuple[str, str]]:
        """Extract character-level samples"""
        samples = []
        
        # Clean text for character learning
        text = text.lower()
        text = ''.join(c if c.isalnum() or c in ' .,!?\n' else ' ' for c in text)
        text = ' '.join(text.split())  # Normalize whitespace
        
        if len(text) < config.min_context_len + 1:
            return samples
        
        # Generate samples with varying context lengths
        for _ in range(min(10, len(text) // 4)):
            ctx_len = random.randint(config.min_context_len, 
                                    min(config.max_context_len, len(text) - 1))
            start = random.randint(0, len(text) - ctx_len - 1)
            
            context = text[start:start + ctx_len]
            target = text[start + ctx_len]
            
            # Encode as bytes
            context_bytes = context.encode('utf-8', errors='replace')
            target_bytes = target.encode('utf-8', errors='replace')
            
            if context_bytes and target_bytes:
                samples.append((context, target))
        
        return samples
    
    def _extract_word_samples(self, text: str, config: PhaseConfig) -> List[Tuple[str, str]]:
        """Extract word-level samples (predict characters within words)"""
        samples = []
        
        # Split into words
        words = text.split()
        if len(words) < 3:
            return samples
        
        # Build context from words
        text_so_far = ""
        for word in words[:50]:  # Limit to first 50 words
            if len(text_so_far) + len(word) + 1 > config.max_context_len:
                break
            
            # Add each character of the word as a sample
            for i in range(len(word)):
                ctx_len = len(text_so_far) + i
                if ctx_len >= config.min_context_len and ctx_len < config.max_context_len:
                    context = text_so_far + word[:i]
                    target = word[i]
                    samples.append((context, target))
            
            text_so_far += word + " "
        
        return samples[:20]  # Limit samples per text
    
    def _extract_sentence_samples(self, text: str, config: PhaseConfig) -> List[Tuple[str, str]]:
        """Extract sentence-level samples"""
        samples = []
        
        # Use longer context windows
        text_bytes = text.encode('utf-8', errors='replace')[:config.max_context_len + 100]
        text = text_bytes.decode('utf-8', errors='replace')
        
        for _ in range(min(5, len(text) // 20)):
            ctx_len = random.randint(config.min_context_len, 
                                    min(config.max_context_len, len(text) - 1))
            start = random.randint(0, max(0, len(text) - ctx_len - 1))
            
            context = text[start:start + ctx_len]
            if start + ctx_len < len(text):
                target = text[start + ctx_len]
                samples.append((context, target))
        
        return samples


# ============================================================================
# MAIN TRAINER
# ============================================================================

class BiologicalTrainer:
    """Main trainer implementing curriculum learning"""
    
    def __init__(self, config: TrainingConfig, resume_path: Optional[str] = None):
        self.config = config
        self.logger = TrainingLogger(config)
        self.viz = BiologicalVisualizer(config)
        self.data_loader = CurriculumDataLoader(config)
        
        # Training state
        self.state = TrainingState(
            config=asdict(config),
            current_phase=TrainingPhase.CHARACTER.value,
            current_epoch=0,
            global_step=0,
            total_tokens=0,
            total_time_seconds=0.0,
        )
        
        # Phase metrics
        self.phase_metrics: Dict[TrainingPhase, PhaseMetrics] = {}
        
        # Model
        self.model = None
        
        # Resume if requested
        if resume_path:
            self._load_checkpoint(resume_path)
        
        # Initialize model
        self._init_model()
        
        # Initialize data
        self.data_loader.initialize()
    
    def _safe_chr(self, char_id: int) -> str:
        """Safely convert character ID to displayable string"""
        if char_id < 0 or char_id > 255:
            return f'[{char_id}]'
        elif char_id == 0:
            return '\\0'
        elif char_id == 9:
            return '\\t'
        elif char_id == 10:
            return '\\n'
        elif char_id == 13:
            return '\\r'
        elif char_id == 32:
            return '␣'  # Space indicator
        elif 33 <= char_id <= 126:
            return chr(char_id)  # Printable ASCII
        elif char_id < 32:
            return f'\\x{char_id:02x}'  # Control characters
        else:
            # Extended ASCII (128-255)
            try:
                return chr(char_id)
            except:
                return f'\\x{char_id:02x}'
    
    def _init_model(self):
        """Initialize the NeuroGen model"""
        try:
            import libneurogen as neurogen
            self.model = neurogen.NeuroGenModel(
                vocab_size=256,
                embedding_dim=256,
                gpu_device=self.config.gpu_device
            )
            self.logger.logger.info("✅ NeuroGen model initialized")
        except ImportError as e:
            self.logger.logger.warning(f"Could not import neurogen: {e}")
            self.logger.logger.warning("Running in simulation mode")
            self.model = None
    
    def _load_checkpoint(self, path: str):
        """Load training state from checkpoint"""
        self.logger.logger.info(f"Loading checkpoint from {path}")
        
        # Load training state
        state_path = Path(path)
        if state_path.exists():
            with open(state_path) as f:
                state_dict = json.load(f)
            
            self.state = TrainingState(**state_dict)
            self.logger.logger.info(f"Resumed at phase {self.state.current_phase}, epoch {self.state.current_epoch}")
        
        # Load model weights
        if self.model:
            try:
                self.model.load_checkpoint(str(state_path.with_suffix('')))
            except Exception as e:
                self.logger.logger.warning(f"Could not load model weights: {e}")
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        phase_name = TrainingPhase(self.state.current_phase).name.lower()
        
        # Update state with current phase metrics (use int keys for JSON compatibility)
        for phase, metrics in self.phase_metrics.items():
            # Convert TrainingPhase to int and handle nested enums
            metrics_dict = asdict(metrics)
            # Convert TrainingPhase enum in metrics to its value
            if 'phase' in metrics_dict and isinstance(metrics_dict['phase'], TrainingPhase):
                metrics_dict['phase'] = metrics_dict['phase'].value
            # Handle epoch_history which contains TrainingPhase
            if 'epoch_history' in metrics_dict:
                for epoch_data in metrics_dict['epoch_history']:
                    if 'phase' in epoch_data and hasattr(epoch_data['phase'], 'value'):
                        epoch_data['phase'] = epoch_data['phase'].value
            self.state.phase_metrics[phase.value] = metrics_dict
        
        # Prepare state for JSON serialization
        state_dict = asdict(self.state)
        
        # Ensure all phase_metrics keys are integers (not TrainingPhase)
        state_dict['phase_metrics'] = {
            int(k) if not isinstance(k, int) else k: v 
            for k, v in state_dict['phase_metrics'].items()
        }
        
        # Save state
        state_filename = f"bio_phase{self.state.current_phase}_{phase_name}_epoch{self.state.current_epoch}.json"
        state_path = checkpoint_dir / state_filename
        
        with open(state_path, 'w') as f:
            json.dump(state_dict, f, indent=2, default=str)
        
        # Save model
        if self.model:
            model_path = state_path.with_suffix('')
            try:
                self.model.save_checkpoint(str(model_path))
            except Exception as e:
                self.logger.logger.warning(f"Could not save model weights: {e}")
        
        # Save best model
        if is_best:
            best_state_path = checkpoint_dir / "bio_best.json"
            with open(best_state_path, 'w') as f:
                json.dump(state_dict, f, indent=2, default=str)
            
            if self.model:
                try:
                    self.model.save_checkpoint(str(best_state_path.with_suffix('')))
                except Exception:
                    pass
            
            self.state.best_checkpoint_path = str(best_state_path)
        
        self.logger.logger.info(f"💾 Checkpoint saved: {state_path}")
    
    def train_epoch(self, phase: TrainingPhase) -> EpochMetrics:
        """Train for one epoch"""
        phase_config = self.config.phases[phase]
        
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_tokens = 0
        
        min_loss = float('inf')
        max_loss = 0.0
        
        # Get samples for this epoch
        samples = list(self.data_loader.get_samples(phase, phase_config.samples_per_epoch))
        random.shuffle(samples)
        
        # Process in batches
        batch_size = phase_config.batch_size
        num_batches = len(samples) // batch_size
        
        progress = tqdm(range(num_batches), desc=f"Epoch {self.state.current_epoch}", 
                       leave=False, ncols=100)
        
        for batch_idx in progress:
            batch_start = batch_idx * batch_size
            batch_samples = samples[batch_start:batch_start + batch_size]
            
            batch_loss = 0.0
            batch_correct = 0
            batch_tokens = 0
            
            for context, target in batch_samples:
                # Encode to bytes
                context_ids = list(context.encode('utf-8', errors='replace'))
                target_id = ord(target) if len(target) == 1 else target.encode('utf-8')[0]
                
                if not context_ids:
                    continue
                
                # Train step
                if self.model:
                    try:
                        loss, accuracy, predicted_id = self.model.train_step(context_ids, [target_id])
                        
                        # CRITICAL: Clamp predicted_id to valid range [0, 255]
                        # The model may output invalid values if not properly initialized
                        if predicted_id < 0 or predicted_id > 255:
                            # Log this anomaly for debugging
                            if self.state.global_step % 100 == 0:
                                self.logger.logger.warning(
                                    f"Invalid predicted_id={predicted_id} at step {self.state.global_step}, clamping to [0,255]"
                                )
                            predicted_id = max(0, min(255, predicted_id))
                        
                        # Safe character conversion
                        predicted = self._safe_chr(predicted_id)
                    except Exception as e:
                        self.logger.logger.warning(f"Train step error: {e}")
                        continue
                else:
                    # Simulation mode
                    loss = random.uniform(1.0, 3.0)
                    accuracy = random.random() * 0.5
                    predicted_id = random.randint(32, 126)  # Printable ASCII only
                    predicted = self._safe_chr(predicted_id)
                
                # Also clamp target_id for safety
                target_id = max(0, min(255, target_id))
                
                correct = (predicted_id == target_id)
                
                batch_loss += loss
                batch_correct += 1 if correct else 0
                batch_tokens += len(context_ids)
                
                # Update visualization
                self.state.global_step += 1
                
                # Log sample
                sample_data = {
                    'input': context[-50:],  # Last 50 chars
                    'expected': target,
                    'predicted': predicted,
                    'correct': correct,
                    'step': self.state.global_step,
                }
                self.state.recent_samples.append(sample_data)
                if len(self.state.recent_samples) > 100:
                    self.state.recent_samples = self.state.recent_samples[-50:]
                
                # Update visualizer
                if self.state.global_step % 10 == 0:
                    self.viz.update(
                        step=self.state.global_step,
                        loss=loss,
                        accuracy=accuracy,
                        phase=phase,
                        sample=sample_data
                    )
            
            # Batch statistics
            if batch_tokens > 0:
                epoch_loss += batch_loss
                epoch_correct += batch_correct
                epoch_total += len(batch_samples)
                epoch_tokens += batch_tokens
                
                min_loss = min(min_loss, batch_loss / len(batch_samples))
                max_loss = max(max_loss, batch_loss / len(batch_samples))
            
            # Update progress bar
            if epoch_total > 0:
                progress.set_postfix({
                    'loss': f'{epoch_loss/epoch_total:.3f}',
                    'acc': f'{epoch_correct/epoch_total:.1%}',
                    'tok': f'{epoch_tokens:,}'
                })
            
            # Generate visualization periodically
            if self.state.global_step % self.config.viz_interval_steps == 0:
                self.viz.generate_dashboard(self.state)
        
        epoch_duration = time.time() - epoch_start
        
        # Compute epoch metrics
        avg_loss = epoch_loss / max(epoch_total, 1)
        avg_accuracy = epoch_correct / max(epoch_total, 1)
        tps = epoch_tokens / max(epoch_duration, 0.001)
        
        metrics = EpochMetrics(
            epoch=self.state.current_epoch,
            phase=phase,
            timestamp=datetime.datetime.now().isoformat(),
            duration_seconds=epoch_duration,
            avg_loss=avg_loss,
            min_loss=min_loss if min_loss != float('inf') else 0,
            max_loss=max_loss,
            avg_accuracy=avg_accuracy,
            char_accuracy=avg_accuracy,
            word_accuracy=0.0,
            tokens_processed=epoch_tokens,
            tokens_per_second=tps,
            samples_processed=epoch_total,
            avg_sparsity=0.0,
            avg_entropy=0.0,
            reward_given=0.0,
            weight_change_magnitude=0.0,
        )
        
        # Update totals
        self.state.total_tokens += epoch_tokens
        self.state.total_time_seconds += epoch_duration
        
        return metrics
    
    def train_phase(self, phase: TrainingPhase) -> bool:
        """Train a complete phase. Returns True if should continue to next phase."""
        phase_config = self.config.phases[phase]
        
        # Initialize phase metrics
        if phase not in self.phase_metrics:
            self.phase_metrics[phase] = PhaseMetrics(
                phase=phase,
                start_time=datetime.datetime.now().isoformat()
            )
        
        pm = self.phase_metrics[phase]
        
        self.logger.log_phase_start(phase, phase_config)
        
        # Set learning rate
        if self.model:
            try:
                self.model.set_decoder_learning_rate(phase_config.learning_rate)
            except Exception:
                pass
        
        epochs_without_improvement = 0
        
        while True:
            self.state.current_epoch += 1
            pm.epochs_completed += 1
            
            # Train epoch
            epoch_metrics = self.train_epoch(phase)
            pm.epoch_history.append(epoch_metrics)
            pm.total_tokens += epoch_metrics.tokens_processed
            pm.total_samples += epoch_metrics.samples_processed
            
            # Track best
            if epoch_metrics.avg_accuracy > pm.best_accuracy:
                pm.best_accuracy = epoch_metrics.avg_accuracy
                epochs_without_improvement = 0
                
                # Save best checkpoint
                if epoch_metrics.avg_accuracy > self.state.best_accuracy:
                    self.state.best_accuracy = epoch_metrics.avg_accuracy
                    self._save_checkpoint(is_best=True)
            else:
                epochs_without_improvement += 1
            
            if epoch_metrics.avg_loss < pm.best_loss:
                pm.best_loss = epoch_metrics.avg_loss
            
            pm.accuracy_history.append(epoch_metrics.avg_accuracy)
            pm.loss_history.append(epoch_metrics.avg_loss)
            pm.final_accuracy = epoch_metrics.avg_accuracy
            
            # Log
            self.logger.log_epoch(epoch_metrics)
            
            # Generate visualization
            self.viz.generate_dashboard(self.state, epoch_metrics)
            self.viz.save_samples_json()
            
            # Save checkpoint
            if pm.epochs_completed % self.config.checkpoint_interval_epochs == 0:
                self._save_checkpoint()
            
            # Check convergence
            converged, reason = pm.is_converged(phase_config)
            if converged:
                pm.end_time = datetime.datetime.now().isoformat()
                self.logger.log_phase_end(phase, pm, reason)
                return True
            
            # Check early stopping
            if epochs_without_improvement >= self.config.patience_epochs:
                self.logger.logger.warning(
                    f"Early stopping: {epochs_without_improvement} epochs without improvement"
                )
                pm.end_time = datetime.datetime.now().isoformat()
                self.logger.log_phase_end(phase, pm, "Early stopping (no improvement)")
                return True
            
            # Check time limit
            if self.config.max_total_hours:
                hours_elapsed = self.state.total_time_seconds / 3600
                if hours_elapsed >= self.config.max_total_hours:
                    self.logger.logger.info(f"Time limit reached ({hours_elapsed:.1f} hours)")
                    return False
    
    def train(self, start_phase: Optional[TrainingPhase] = None, 
              end_phase: Optional[TrainingPhase] = None):
        """Main training loop"""
        start_time = time.time()
        
        self.logger.logger.info("=" * 80)
        self.logger.logger.info("🧠 NeuroGen 2.5 Biological Curriculum Training")
        self.logger.logger.info("=" * 80)
        
        # Determine phases to train
        if start_phase is None:
            start_phase = TrainingPhase(self.state.current_phase)
        if end_phase is None:
            end_phase = TrainingPhase.SENTENCE
        
        phases_to_train = [p for p in TrainingPhase 
                         if start_phase.value <= p.value <= end_phase.value]
        
        self.logger.logger.info(f"Training phases: {[p.name for p in phases_to_train]}")
        
        try:
            for phase in phases_to_train:
                self.state.current_phase = phase.value
                
                should_continue = self.train_phase(phase)
                
                if not should_continue:
                    break
                
                # Reset epoch counter for new phase
                self.state.current_epoch = 0
            
            # Final save
            self._save_checkpoint()
            self.logger.save_summary(self.state)
            
        except KeyboardInterrupt:
            self.logger.logger.warning("Training interrupted by user")
            self._save_checkpoint()
            self.logger.save_summary(self.state)
        
        except Exception as e:
            self.logger.logger.error(f"Training error: {e}")
            traceback.print_exc()
            self._save_checkpoint()
            raise
        
        finally:
            total_time = time.time() - start_time
            self.logger.logger.info("=" * 80)
            self.logger.logger.info("🎉 Training Complete!")
            self.logger.logger.info(f"   Total time: {total_time/3600:.2f} hours")
            self.logger.logger.info(f"   Total tokens: {self.state.total_tokens:,}")
            self.logger.logger.info(f"   Best accuracy: {self.state.best_accuracy:.2%}")
            self.logger.logger.info("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NeuroGen 2.5 Biological Curriculum Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from scratch, all phases
  python train_biological.py --phase all

  # Train only character phase for 5 epochs
  python train_biological.py --phase 1 --max-epochs 5

  # Resume from checkpoint
  python train_biological.py --resume checkpoints/bio_phase1_character_epoch3.json

  # Quick test mode
  python train_biological.py --test
        """
    )
    
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--phase", type=str, default="all",
                       help="Phase to train: 1 (character), 2 (word), 3 (sentence), or 'all'")
    parser.add_argument("--max-epochs", type=int, default=None,
                       help="Maximum epochs per phase (overrides config)")
    parser.add_argument("--max-hours", type=float, default=None,
                       help="Maximum total training hours")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--test", action="store_true",
                       help="Quick test mode (reduced samples)")
    parser.add_argument("--viz-interval", type=int, default=100,
                       help="Steps between visualization updates")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        gpu_device=args.gpu,
        max_total_hours=args.max_hours,
        viz_interval_steps=args.viz_interval,
    )
    
    # Test mode adjustments
    if args.test:
        print("🧪 Running in TEST MODE")
        for phase_config in config.phases.values():
            phase_config.samples_per_epoch = 1000
            phase_config.max_epochs = 2
            phase_config.min_epochs = 1
        config.viz_interval_steps = 50
        config.checkpoint_interval_epochs = 1
    
    # Max epochs override
    if args.max_epochs:
        for phase_config in config.phases.values():
            phase_config.max_epochs = args.max_epochs
    
    # Determine start/end phases
    start_phase = None
    end_phase = None
    
    if args.phase != "all":
        try:
            phase_num = int(args.phase)
            start_phase = TrainingPhase(phase_num)
            end_phase = TrainingPhase(phase_num)
        except ValueError:
            print(f"Invalid phase: {args.phase}")
            sys.exit(1)
    
    # Print config
    print("\n" + "=" * 80)
    print("🧠 NeuroGen 2.5 - Biological Curriculum Training")
    print("=" * 80)
    print(f"GPU Device: {config.gpu_device}")
    print(f"Phases: {args.phase}")
    print(f"Max Hours: {args.max_hours or 'unlimited'}")
    print(f"Resume: {args.resume or 'No'}")
    print(f"Visualization: {config.viz_dir}")
    print(f"Checkpoints: {config.checkpoint_dir}")
    print("=" * 80 + "\n")
    
    # Create and run trainer
    trainer = BiologicalTrainer(config, resume_path=args.resume)
    trainer.train(start_phase=start_phase, end_phase=end_phase)


if __name__ == "__main__":
    main()
