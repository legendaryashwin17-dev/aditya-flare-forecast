"""
Utility functions for reproducibility, logging, and checkpointing.
"""

import os
import sys
import random
import logging
from typing import Dict, Optional, Any

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(log_level: str = "INFO",
                  log_file: Optional[str] = None):
    """Configure logging with console and optional file output."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers
    )


def save_checkpoint(state: Dict, filepath: str):
    """Save model checkpoint with metrics and training state."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    logging.getLogger(__name__).info(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str, model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None
                    ) -> Dict[str, Any]:
    """Load model checkpoint and optionally restore optimizer state."""
    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logging.getLogger(__name__).info(
        f"Checkpoint loaded from {filepath} (epoch {checkpoint.get('epoch', '?')})"
    )
    return checkpoint
