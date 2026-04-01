from __future__ import annotations

from typing import Optional

try:
    from openbayestool import log_param, log_metric, clear_metric, clear_param
    _openbayestool_available = True
except ImportError:
    _openbayestool_available = False

import lightning.pytorch as pl


class EpochMetricsPrinter(pl.Callback):
    def __init__(self, log_params: Optional[dict] = None, console: bool = True, openbayestool: bool = True):
        """
        Args:
            log_params: hyperparameters to log once at the start (e.g. lr, batch_size).
            console: whether to print metrics to stdout each epoch.
            openbayestool: whether to use openbayestool logging if available.
        """
        self._log_params = log_params or {}
        self._console = console
        self._use_openbayestool = openbayestool and _openbayestool_available
        self._cleared_metrics: set = set()

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: ARG002
        if self._log_params and trainer.logger:
            trainer.logger.log_hyperparams(self._log_params)
        if self._use_openbayestool and trainer.is_global_zero:
            for k in self._log_params:
                clear_param(k)  # type: ignore
            for k, v in self._log_params.items():
                log_param(k, v)  # type: ignore

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: ARG002
        metrics = {k: v for k, v in trainer.callback_metrics.items() if "train" in k}
        if not metrics:
            return
        if self._console and trainer.is_global_zero:
            print(f"[Epoch {trainer.current_epoch}] " + "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
        if self._use_openbayestool and trainer.is_global_zero:
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
        if self._console and trainer.is_global_zero:
            print(f"[Epoch {trainer.current_epoch}] " + "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
        if self._use_openbayestool and trainer.is_global_zero:
            for k, v in metrics.items():
                if k not in self._cleared_metrics:
                    clear_metric(k)  # type: ignore
                    self._cleared_metrics.add(k)
                log_metric(k, float(v))  # type: ignore
