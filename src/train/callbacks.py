"""Training callbacks for early stopping, checkpointing, and logging."""
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .trainer import Trainer

from .trainer import Callback


class EarlyStopping(Callback):
    """Early stopping callback to stop training when validation metric stops improving.
    
    Example:
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
            min_delta=0.001,
        )
        trainer.train(epochs=50, callbacks=[early_stop])
    """
    
    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        mode: str = "min",
        min_delta: float = 0.0,
        verbose: bool = True,
    ):
        """
        Args:
            monitor: Metric to monitor ("val_loss", "val_acc", "val_f1")
            patience: Number of epochs with no improvement to wait
            mode: "min" for loss, "max" for accuracy/F1
            min_delta: Minimum change to qualify as improvement
            verbose: Print when stopping
        """
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.counter = 0
        self.best_epoch = 0
    
    def on_train_begin(self, trainer: "Trainer"):
        self.best_value = float('inf') if self.mode == "min" else float('-inf')
        self.counter = 0
        self.best_epoch = 0
    
    def on_train_end(self, trainer: "Trainer"):
        pass
    
    def on_epoch_begin(self, trainer: "Trainer", epoch: int):
        pass
    
    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: Dict[str, float]):
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if self._is_improvement(current):
            self.best_value = current
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"  EarlyStopping: No improvement for {self.counter}/{self.patience} epochs "
                    f"(best {self.monitor}: {self.best_value:.4f} at epoch {self.best_epoch + 1})"
                )
            
            if self.counter >= self.patience:
                trainer.stop_training = True
    
    def _is_improvement(self, current: float) -> bool:
        if self.mode == "min":
            return current < self.best_value - self.min_delta
        else:
            return current > self.best_value + self.min_delta


class ModelCheckpoint(Callback):
    """Save model checkpoints during training.
    
    Example:
        checkpoint = ModelCheckpoint(
            filepath="checkpoints/model_{epoch}_{val_f1:.4f}.pt",
            monitor="val_f1",
            mode="max",
            save_best_only=True,
        )
        trainer.train(epochs=20, callbacks=[checkpoint])
    """
    
    def __init__(
        self,
        filepath: str | Path,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        verbose: bool = True,
    ):
        """
        Args:
            filepath: Path template for saving. Can include {epoch}, {val_loss}, etc.
            monitor: Metric to monitor for best model
            mode: "min" for loss, "max" for accuracy/F1
            save_best_only: Only save when monitored metric improves
            verbose: Print when saving
        """
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.best_path: Optional[Path] = None
    
    def on_train_begin(self, trainer: "Trainer"):
        self.best_value = float('inf') if self.mode == "min" else float('-inf')
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def on_train_end(self, trainer: "Trainer"):
        if self.verbose and self.best_path:
            print(f"Best model saved to: {self.best_path}")
    
    def on_epoch_begin(self, trainer: "Trainer", epoch: int):
        pass
    
    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: Dict[str, float]):
        current = logs.get(self.monitor)
        if current is None:
            return
        
        # Format filepath with metrics
        filepath = str(self.filepath).format(
            epoch=epoch + 1,
            **logs,
        )
        filepath = Path(filepath)
        
        if self.save_best_only:
            if self._is_improvement(current):
                self.best_value = current
                self.best_path = filepath
                trainer.save_checkpoint(filepath)
                if self.verbose:
                    print(f"  Checkpoint: Saved best model ({self.monitor}={current:.4f}) to {filepath}")
        else:
            trainer.save_checkpoint(filepath)
            if self._is_improvement(current):
                self.best_value = current
                self.best_path = filepath
            if self.verbose:
                print(f"  Checkpoint: Saved to {filepath}")
    
    def _is_improvement(self, current: float) -> bool:
        if self.mode == "min":
            return current < self.best_value
        else:
            return current > self.best_value


class LearningRateLogger(Callback):
    """Log learning rate changes during training."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.last_lr: Optional[float] = None
    
    def on_train_begin(self, trainer: "Trainer"):
        self.last_lr = trainer.optimizer.param_groups[0]['lr']
    
    def on_train_end(self, trainer: "Trainer"):
        pass
    
    def on_epoch_begin(self, trainer: "Trainer", epoch: int):
        pass
    
    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: Dict[str, float]):
        current_lr = trainer.optimizer.param_groups[0]['lr']
        if self.verbose and self.last_lr and current_lr != self.last_lr:
            print(f"  LR changed: {self.last_lr:.2e} → {current_lr:.2e}")
        self.last_lr = current_lr


class HistoryLogger(Callback):
    """Save training history to file after each epoch."""
    
    def __init__(self, filepath: str | Path, save_every: int = 1):
        """
        Args:
            filepath: Path to save history JSON
            save_every: Save every N epochs
        """
        self.filepath = Path(filepath)
        self.save_every = save_every
    
    def on_train_begin(self, trainer: "Trainer"):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def on_train_end(self, trainer: "Trainer"):
        self._save(trainer)
    
    def on_epoch_begin(self, trainer: "Trainer", epoch: int):
        pass
    
    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: Dict[str, float]):
        if (epoch + 1) % self.save_every == 0:
            self._save(trainer)
    
    def _save(self, trainer: "Trainer"):
        import json
        with open(self.filepath, "w") as f:
            json.dump(trainer.history.to_dict(), f, indent=2)


class GradientMonitor(Callback):
    """Monitor gradient norms during training for debugging."""
    
    def __init__(self, verbose: bool = True, log_every: int = 100):
        """
        Args:
            verbose: Print gradient statistics
            log_every: Log every N batches (approximate via epochs)
        """
        self.verbose = verbose
        self.log_every = log_every
        self.gradient_norms: list = []
    
    def on_train_begin(self, trainer: "Trainer"):
        self.gradient_norms = []
    
    def on_train_end(self, trainer: "Trainer"):
        if self.verbose and self.gradient_norms:
            import statistics
            print(f"\nGradient Statistics:")
            print(f"  Mean norm: {statistics.mean(self.gradient_norms):.4f}")
            print(f"  Max norm: {max(self.gradient_norms):.4f}")
            print(f"  Min norm: {min(self.gradient_norms):.4f}")
    
    def on_epoch_begin(self, trainer: "Trainer", epoch: int):
        pass
    
    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: Dict[str, float]):
        # Compute gradient norm
        total_norm = 0.0
        for p in trainer.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self.gradient_norms.append(total_norm)
        
        if self.verbose:
            print(f"  Gradient norm: {total_norm:.4f}")
