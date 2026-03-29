from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

try:
    from openbayestool import log_param, log_metric, clear_metric, clear_param
    _openbayestool_available = True
except ImportError:
    _openbayestool_available = False

import lightning.pytorch as pl


class EpochMetricsPrinter(pl.Callback):
    def __init__(self, log_params: Optional[dict] = None, console: bool = True):
        """
        Args:
            log_params: hyperparameters to log once at the start (e.g. lr, batch_size).
            console: whether to print metrics to stdout each epoch.
        """
        self._log_params = log_params or {}
        self._console = console
        self._cleared_metrics: set = set()

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: ARG002
        if self._log_params and trainer.logger:
            trainer.logger.log_hyperparams(self._log_params)
            log_dir = trainer.logger.log_dir
            if log_dir:
                args_path = Path(log_dir) / "args.json"
                args_path.parent.mkdir(parents=True, exist_ok=True)
                with open(args_path, "w") as f:
                    json.dump(self._log_params, f, indent=2)
        if _openbayestool_available:
            for k in self._log_params:
                clear_param(k)  # type: ignore
            for k, v in self._log_params.items():
                log_param(k, v)  # type: ignore

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: ARG002
        metrics = {k: v for k, v in trainer.callback_metrics.items() if "train" in k}
        if not metrics:
            return
        if self._console:
            print(f"[Epoch {trainer.current_epoch}] " + "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
        if _openbayestool_available:
            for k, v in metrics.items():
                if k not in self._cleared_metrics:
                    clear_metric(k)  # type: ignore
                    self._cleared_metrics.add(k)
                log_metric(k, float(v))  # type: ignore

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: ARG002
        if trainer.sanity_checking:
            return
        metrics = {k: v for k, v in trainer.callback_metrics.items() if "val" in k}
        if not metrics:
            return
        if self._console:
            print(f"[Epoch {trainer.current_epoch}] " + "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
        if _openbayestool_available:
            for k, v in metrics.items():
                if k not in self._cleared_metrics:
                    clear_metric(k)  # type: ignore
                    self._cleared_metrics.add(k)
                log_metric(k, float(v))  # type: ignore
