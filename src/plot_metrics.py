import pandas as pd
import matplotlib.pyplot as plt

LOG_DIR = "lightning_logs/version_1"

with open(f"{LOG_DIR}/hparams.yaml") as f:
    hparams_text = f.read()

df = pd.read_csv(f"{LOG_DIR}/metrics.csv")

epoch_df = df.groupby("epoch").mean(numeric_only=True)

val_cols = ["val_acc1", "val_acc5", "val_knn_acc1", "val_knn_acc5", "val_loss"]
train_cols = ["train_nce_loss_epoch", "train_class_loss_epoch", "train_acc1_epoch", "train_acc5_epoch"]

available_val = [c for c in val_cols if c in epoch_df.columns and epoch_df[c].notna().any()]
available_train = [c for c in train_cols if c in epoch_df.columns and epoch_df[c].notna().any()]

def plot(ax, data, label):
    ax.plot(data.index, data.values, linewidth=0.8, label=label)

val_loss_cols   = [c for c in available_val   if "loss" in c]
val_acc_cols    = [c for c in available_val   if "acc"  in c]
train_loss_cols = [c for c in available_train if "loss" in c]
train_acc_cols  = [c for c in available_train if "acc"  in c]

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.1], hspace=0.35, wspace=0.35)
ax_vl = fig.add_subplot(gs[0, 0])
ax_va = fig.add_subplot(gs[0, 1])
ax_tl = fig.add_subplot(gs[1, 0])
ax_ta = fig.add_subplot(gs[1, 1])
ax_hp = fig.add_subplot(gs[:, 2])

for col in val_loss_cols:
    plot(ax_vl, epoch_df[col].dropna(), col)
ax_vl.set_title("Validation loss")
ax_vl.set_xlabel("Epoch")
ax_vl.legend()
ax_vl.grid(True, alpha=0.3)

for col in val_acc_cols:
    plot(ax_va, epoch_df[col].dropna(), col)
ax_va.set_title("Validation accuracy")
ax_va.set_xlabel("Epoch")
ax_va.legend()
ax_va.grid(True, alpha=0.3)

for col in train_loss_cols:
    plot(ax_tl, epoch_df[col].dropna(), col)
ax_tl.set_title("Train loss")
ax_tl.set_xlabel("Epoch")
ax_tl.legend()
ax_tl.grid(True, alpha=0.3)

for col in train_acc_cols:
    plot(ax_ta, epoch_df[col].dropna(), col)
ax_ta.set_title("Train accuracy")
ax_ta.set_xlabel("Epoch")
ax_ta.legend()
ax_ta.grid(True, alpha=0.3)

ax_hp.axis("off")
ax_hp.text(
    0.05, 0.95, hparams_text,
    transform=ax_hp.transAxes, va="top", ha="left",
    fontsize=8, family="monospace",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
)
ax_hp.set_title("Hyperparameters")

plt.tight_layout()
plt.savefig("metrics.png", dpi=150)
plt.show()
print("Saved metrics.png")
