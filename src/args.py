import argparse


def setup_args(parser: argparse.ArgumentParser) -> None:
    """Register all training arguments. Called by both main.py and launch_distributed.py."""
    parser.add_argument("--method", type=str, default="simclr", choices=["simclr", "byol"])
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100", "tinyimagenet"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.1, help="SimCLR only")
    parser.add_argument("--proj_output_dim", type=int, default=128)
    parser.add_argument("--proj_hidden_dim", type=int, default=2048)
    parser.add_argument("--pred_hidden_dim", type=int, default=512, help="BYOL only")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_every_n_epochs", type=int, default=1)
    parser.add_argument("--precision", type=str, default="32", choices=["32", "16-mixed", "bf16-mixed"],
                        help="Training precision. Use 32 for GPUs without tensor cores (e.g. GTX 1660)")
    parser.add_argument("--no_console_log", action="store_true", help="Disable printing metrics to stdout")
    parser.add_argument("--openbayestool", action="store_true", help="Enable openbayestool logging if available")
