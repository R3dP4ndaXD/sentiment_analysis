"""Training utilities and Trainer class for sentiment classification."""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from ..evaluate.metrics import compute_metrics


@dataclass
class TrainHistory:
    """Stores training history for plotting and analysis."""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    train_f1: List[float] = field(default_factory=list)
    val_f1: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_acc": self.train_acc,
            "val_acc": self.val_acc,
            "train_f1": self.train_f1,
            "val_f1": self.val_f1,
            "learning_rates": self.learning_rates,
            "epoch_times": self.epoch_times,
        }


class Trainer:
    """Training loop manager for sentiment classification models.
    
    Handles:
    - Training and validation loops
    - Optimizer and scheduler configuration
    - Metrics tracking (loss, accuracy, F1)
    - Checkpointing and early stopping via callbacks
    - Device management (CPU/GPU)
    
    Example:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer="adamw",
            lr=1e-3,
            device="cuda",
        )
        history = trainer.train(epochs=20, callbacks=[early_stopping])
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: str = "adamw",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        scheduler: Optional[str] = "plateau",
        scheduler_patience: int = 3,
        scheduler_factor: float = 0.5,
        device: Optional[str] = None,
        criterion: Optional[nn.Module] = None,
        gradient_clip: Optional[float] = 1.0,
    ):
        """
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer name ("adam", "adamw", "sgd")
            lr: Learning rate
            weight_decay: L2 regularization weight
            scheduler: LR scheduler ("plateau", "cosine", None)
            scheduler_patience: Patience for ReduceLROnPlateau
            scheduler_factor: Factor for ReduceLROnPlateau
            device: Device to train on ("cuda", "cpu", or None for auto)
            criterion: Loss function (default: CrossEntropyLoss)
            gradient_clip: Max gradient norm for clipping (None to disable)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        
        # Device setup
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        
        # Loss function
        self.criterion = criterion or nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = self._create_optimizer(optimizer, lr, weight_decay)
        
        # Scheduler
        self.scheduler = self._create_scheduler(
            scheduler, scheduler_patience, scheduler_factor
        )
        self.scheduler_name = scheduler
        
        # Training state
        self.history = TrainHistory()
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.stop_training = False
    
    def _create_optimizer(
        self, 
        name: str, 
        lr: float, 
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        """Create optimizer by name."""
        optimizers = {
            "adam": Adam,
            "adamw": AdamW,
            "sgd": lambda params, lr, weight_decay: SGD(
                params, lr=lr, weight_decay=weight_decay, momentum=0.9
            ),
        }
        
        if name.lower() not in optimizers:
            raise ValueError(f"Unknown optimizer: {name}. Available: {list(optimizers.keys())}")
        
        return optimizers[name.lower()](
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    
    def _create_scheduler(
        self,
        name: Optional[str],
        patience: int,
        factor: float,
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if name is None:
            return None
        
        if name.lower() == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=patience,
                factor=factor,
            )
        elif name.lower() == "cosine":
            # Will be re-initialized when we know total epochs
            return None
        else:
            raise ValueError(f"Unknown scheduler: {name}")
    
    def _train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch.
        
        Returns:
            Tuple of (average loss, metrics dict)
        """
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
   
        for batch in self.train_loader:
            # Unpack batch: (token_ids, labels, lengths)
            # where:
            #   token_ids: Input token indices (batch, seq_len)
            #   lengths: Original sequence lengths (batch,)
            
            token_ids, labels, lengths = batch
            token_ids = token_ids.to(self.device)
            labels = labels.to(self.device)
            lengths = lengths.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(token_ids, lengths)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip is not None:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.gradient_clip
                )
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
        
        avg_loss = total_loss / len(self.train_loader.dataset)
        metrics = compute_metrics(all_labels, all_preds)
        
        return avg_loss, metrics
    
    @torch.no_grad()
    def _validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch.
        
        Returns:
            Tuple of (average loss, metrics dict)
        """
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch in self.val_loader:
            token_ids, labels, lengths = batch
            token_ids = token_ids.to(self.device)
            labels = labels.to(self.device)
            lengths = lengths.to(self.device)
            
            logits = self.model(token_ids, lengths)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
        
        avg_loss = total_loss / len(self.val_loader.dataset)
        metrics = compute_metrics(all_labels, all_preds)
        
        return avg_loss, metrics
    
    def train(
        self,
        epochs: int,
        callbacks: Optional[List["Callback"]] = None,
        verbose: bool = True,
    ) -> TrainHistory:
        """Run full training loop.
        
        Args:
            epochs: Number of epochs to train
            callbacks: List of callback objects
            verbose: Print progress
        
        Returns:
            TrainHistory with metrics for each epoch
        """
        callbacks = callbacks or []
        
        # Initialize cosine scheduler if requested
        if self.scheduler_name and self.scheduler_name.lower() == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        # Callback: on_train_begin
        for cb in callbacks:
            cb.on_train_begin(self)
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Callback: on_epoch_begin
            for cb in callbacks:
                cb.on_epoch_begin(self, epoch)
            
            # Training
            train_loss, train_metrics = self._train_epoch()
            
            # Validation
            val_loss, val_metrics = self._validate_epoch()
            
            # Update scheduler
            lr_before = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log LR change (replaces deprecated verbose=True)
            if verbose and current_lr != lr_before:
                print(f"  >> Learning rate reduced: {lr_before:.2e} → {current_lr:.2e}")
            
            # Record history
            epoch_time = time.time() - epoch_start
            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(val_loss)
            self.history.train_acc.append(train_metrics["accuracy"])
            self.history.val_acc.append(val_metrics["accuracy"])
            self.history.train_f1.append(train_metrics["f1"])
            self.history.val_f1.append(val_metrics["f1"])
            self.history.learning_rates.append(current_lr)
            self.history.epoch_times.append(epoch_time)
            
            # Track best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            if val_metrics["f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1"]
            
            # Print progress
            if verbose:
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Train Acc: {train_metrics['accuracy']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"Train F1: {train_metrics['f1']:.4f} | "
                    f"Val F1: {val_metrics['f1']:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {epoch_time:.1f}s"
                )
            
            # Callback: on_epoch_end
            logs = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_metrics["accuracy"],
                "val_acc": val_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "val_f1": val_metrics["f1"],
                "lr": current_lr,
            }
            for cb in callbacks:
                cb.on_epoch_end(self, epoch, logs)
            
            # Check early stopping
            if self.stop_training:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Callback: on_train_end
        for cb in callbacks:
            cb.on_train_end(self)
        
        return self.history
    
    @torch.no_grad()
    def predict(
        self, 
        data_loader: DataLoader,
    ) -> Tuple[List[int], List[int], List[List[float]]]:
        """Generate predictions for a dataset.
        
        Args:
            data_loader: DataLoader to predict on
        
        Returns:
            Tuple of (predictions, labels, probabilities)
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in data_loader:
            token_ids, labels, lengths = batch
            token_ids = token_ids.to(self.device)
            lengths = lengths.to(self.device)
            
            logits = self.model(token_ids, lengths)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())
        
        return all_preds, all_labels, all_probs
    
    def save_checkpoint(self, path: Path | str, include_optimizer: bool = True):
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            include_optimizer: Include optimizer state for resuming training
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_f1": self.best_val_f1,
            "history": self.history.to_dict(),
        }
        
        if include_optimizer:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
            if self.scheduler is not None:
                checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path | str, load_optimizer: bool = True):
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint
            load_optimizer: Load optimizer state for resuming training
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        self.best_val_f1 = checkpoint.get("best_val_f1", 0.0)
        
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if load_optimizer and "scheduler_state_dict" in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Restore history
        if "history" in checkpoint:
            hist = checkpoint["history"]
            self.history = TrainHistory(
                train_loss=hist.get("train_loss", []),
                val_loss=hist.get("val_loss", []),
                train_acc=hist.get("train_acc", []),
                val_acc=hist.get("val_acc", []),
                train_f1=hist.get("train_f1", []),
                val_f1=hist.get("val_f1", []),
                learning_rates=hist.get("learning_rates", []),
                epoch_times=hist.get("epoch_times", []),
            )


class Callback:
    """Base class for training callbacks."""
    
    def on_train_begin(self, trainer: Trainer):
        """Called at the start of training."""
        pass
    
    def on_train_end(self, trainer: Trainer):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, trainer: Trainer, epoch: int):
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: Dict[str, float]):
        """Called at the end of each epoch."""
        pass
