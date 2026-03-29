try:
    from openbayestool import log_param, log_metric
    _openbayestool_available = True
except ImportError:
    _openbayestool_available = False

import lightning.pytorch as pl


class EpochMetricsPrinter(pl.Callback):
    def __init__(self, log_params: dict | None = None):
        """
        Args:
            log_params: hyperparameters to log once at the start (e.g. lr, batch_size).
        """
        self._log_params = log_params or {}

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: ARG002
        if _openbayestool_available:
            for k, v in self._log_params.items():
                log_param(k, v)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: ARG002
        metrics = {k: v for k, v in trainer.callback_metrics.items() if "train" in k}
        if not metrics:
            return
        print(f"[Epoch {trainer.current_epoch}] " + "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
        if _openbayestool_available:
            for k, v in metrics.items():
                log_metric(k, float(v))

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: ARG002
        if trainer.sanity_checking:
            return
        metrics = {k: v for k, v in trainer.callback_metrics.items() if "val" in k}
        if not metrics:
            return
        print(f"[Epoch {trainer.current_epoch}] " + "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
        if _openbayestool_available:
            for k, v in metrics.items():
                log_metric(k, float(v))
