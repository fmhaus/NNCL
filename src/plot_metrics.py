import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--path", default="lightning_logs/version_1")
parser.add_argument("--path2", default=None)
args = parser.parse_args()

val_cols   = ["val_acc1", "val_acc5", "val_knn_acc1", "val_knn_acc5", "val_loss"]
train_cols = ["train_nce_loss_epoch", "train_class_loss_epoch", "train_acc1_epoch", "train_acc5_epoch"]


def load(log_dir):
    with open(f"{log_dir}/hparams.yaml") as f:
        hparams_text = f.read()
    df = pd.read_csv(f"{log_dir}/metrics.csv")
    epoch_df = df.groupby("epoch").mean(numeric_only=True)
    avail_val   = [c for c in val_cols   if c in epoch_df.columns and epoch_df[c].notna().any()]
    avail_train = [c for c in train_cols if c in epoch_df.columns and epoch_df[c].notna().any()]
    return epoch_df, avail_val, avail_train, hparams_text


def plot_series(ax, epoch_df, cols, color_map, linestyle, alpha=1.0, label_suffix=""):
    for col in cols:
        data = epoch_df[col].dropna()
        ax.plot(
            data.index, data.values,
            linewidth=0.8,
            linestyle=linestyle,
            color=color_map[col],
            alpha=alpha,
            label=col + label_suffix,
        )


epoch_df1, avail_val1, avail_train1, hparams1 = load(args.path)

hparams_text = hparams1
if args.path2:
    epoch_df2, avail_val2, avail_train2, hparams2 = load(args.path2)
    epoch_df1.index = epoch_df1.index / epoch_df1.index.max() * 100
    epoch_df2.index = epoch_df2.index / epoch_df2.index.max() * 100
    hparams_text = f"--- {args.path} ---\n{hparams1}\n--- {args.path2} ---\n{hparams2}"

val_loss_cols   = [c for c in avail_val1   if "loss" in c]
val_acc_cols    = [c for c in avail_val1   if "acc"  in c]
train_loss_cols = [c for c in avail_train1 if "loss" in c]
train_acc_cols  = [c for c in avail_train1 if "acc"  in c]

prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
all_cols = val_loss_cols + val_acc_cols + train_loss_cols + train_acc_cols
color_map = {col: prop_cycle[i % len(prop_cycle)] for i, col in enumerate(all_cols)}

fig = plt.figure(figsize=(18, 10), layout="constrained")
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.2], hspace=0.35, wspace=0.35)
ax_vl = fig.add_subplot(gs[0, 0])
ax_va = fig.add_subplot(gs[0, 1])
ax_tl = fig.add_subplot(gs[1, 0])
ax_ta = fig.add_subplot(gs[1, 1])
ax_hp = fig.add_subplot(gs[:, 2])

suffix1 = f" ({args.path})" if args.path2 else ""
plot_series(ax_vl, epoch_df1, val_loss_cols,   color_map, linestyle="-", label_suffix=suffix1)
plot_series(ax_va, epoch_df1, val_acc_cols,    color_map, linestyle="-", label_suffix=suffix1)
plot_series(ax_tl, epoch_df1, train_loss_cols, color_map, linestyle="-", label_suffix=suffix1)
plot_series(ax_ta, epoch_df1, train_acc_cols,  color_map, linestyle="-", label_suffix=suffix1)

if args.path2:
    plot_series(ax_vl, epoch_df2, [c for c in val_loss_cols   if c in avail_val2],   color_map, linestyle="--", alpha=0.7, label_suffix=f" ({args.path2})")
    plot_series(ax_va, epoch_df2, [c for c in val_acc_cols    if c in avail_val2],   color_map, linestyle="--", alpha=0.7, label_suffix=f" ({args.path2})")
    plot_series(ax_tl, epoch_df2, [c for c in train_loss_cols if c in avail_train2], color_map, linestyle="--", alpha=0.7, label_suffix=f" ({args.path2})")
    plot_series(ax_ta, epoch_df2, [c for c in train_acc_cols  if c in avail_train2], color_map, linestyle="--", alpha=0.7, label_suffix=f" ({args.path2})")

x_label = "Training %" if args.path2 else "Epoch"
for ax, title in [(ax_vl, "Validation loss"), (ax_va, "Validation accuracy"),
                  (ax_tl, "Train loss"), (ax_ta, "Train accuracy")]:
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

ax_hp.axis("off")
ax_hp.text(
    0.05, 0.95, hparams_text,
    transform=ax_hp.transAxes, va="top", ha="left",
    fontsize=7, family="monospace",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
)
ax_hp.set_title("Hyperparameters")

if args.path2:
    other = args.path2.rstrip("/").split("/")[-1]
    save_path = f"{args.path}/metrics_vs_{other}.png"
else:
    save_path = f"{args.path}/metrics.png"
plt.savefig(save_path, dpi=150)
plt.show()
print(f"Saved {save_path}")
