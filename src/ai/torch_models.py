"""PyTorch nn.Module definitions for Pi-side inference.

Mirrors classes defined locally inside trainer scripts (training/train_*.py)
so that ``state_dict`` checkpoints produced by the trainer can be reloaded on
the Pi without depending on the trainer module being importable.

This module is imported lazily from ``src.ai.predictor`` only when a branch
that actually needs torch is reached (e.g. the lstm branch in
``predict_multi``). A Pi runtime without torch installed never executes the
import path.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """Multi-horizon LSTM forecaster.

    Architecture identical to ``LSTMForecaster`` in ``training/train_lstm.py``
    (currently lines 181-198). Layer attribute names (``lstm``, ``dropout``,
    ``head``) must match exactly so that ``load_state_dict`` accepts the
    trained checkpoint without missing/unexpected-key errors.

    ``sequence_len`` is accepted as a kwarg purely so that
    ``LSTMForecaster(**architecture_config)`` works directly with the
    architecture_config dict produced by the trainer (it stores
    ``sequence_len`` as a hyperparameter even though the layer itself is
    sequence-length-agnostic at construction time).
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        n_horizons: int,
        sequence_len: int = 0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, n_horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(self.dropout(last))
