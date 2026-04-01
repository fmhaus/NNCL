import argparse
from pathlib import Path

import omegaconf
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from solo.methods import SimCLR, BYOL
from solo.data.pretrain_dataloader import (
    NCropAugmentation,
    FullTransformPipeline,
    prepare_dataloader,
    prepare_datasets,
    dataset_with_index,
)
from solo.data.classification_dataloader import prepare_datasets as prepare_cls_datasets
from logger import EpochMetricsPrinter

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

TINYIMAGENET_MEAN = (0.485, 0.456, 0.406)
TINYIMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(dataset: str) -> transforms.Compose:
    if dataset == "cifar100":
        return transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
        ])
    else:  # tinyimagenet
        return transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=TINYIMAGENET_MEAN, std=TINYIMAGENET_STD),
        ])


def build_val_transform(dataset: str) -> transforms.Compose:
    if dataset == "cifar100":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
        ])
    else:  # tinyimagenet
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=TINYIMAGENET_MEAN, std=TINYIMAGENET_STD),
        ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSL pretraining")
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
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_every_n_epochs", type=int, default=1)
    parser.add_argument("--precision", type=str, default="32", choices=["32", "16-mixed", "bf16-mixed"],
                        help="Training precision. Use 32 for GPUs without tensor cores (e.g. GTX 1660)")
    parser.add_argument("--no_console_log", action="store_true", help="Disable printing metrics to stdout")
    parser.add_argument("--openbayestool", action="store_true", help="Enable openbayestool logging if available")
    return parser.parse_args()


def build_cfg(args: argparse.Namespace) -> omegaconf.DictConfig:
    num_classes = 100 if args.dataset == "cifar100" else 200
    # solo-learn uses "imagenet" path for ImageFolder-style datasets
    solo_dataset = args.dataset if args.dataset == "cifar100" else "imagenet"

    cfg = omegaconf.OmegaConf.create({
        "method": args.method,
        "backbone": {
            "name": "resnet18",
            "kwargs": {},
        },
        "data": {
            "dataset": solo_dataset,
            "num_classes": num_classes,
            "num_large_crops": 2,
            "num_small_crops": 0,
        },
        "max_epochs": args.max_epochs,
        "accumulate_grad_batches": 1,
        "optimizer": {
            "name": "sgd",
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "classifier_lr": 0.1,
            "exclude_bias_n_norm_wd": True,
            "kwargs": {"momentum": 0.9},
        },
        "scheduler": {
            "name": "warmup_cosine",
            "min_lr": 0.0,
            "warmup_start_lr": 0.003,
            "warmup_epochs": 10,
            "lr_decay_steps": None,
            "interval": "step",
        },
        "knn_eval": {
            "enabled": True,
            "k": 20,
            "distance_func": "euclidean",
        },
        "performance": {
            "disable_channel_last": False,
        },
        "method_kwargs": {
            "proj_output_dim": args.proj_output_dim,
            "proj_hidden_dim": args.proj_hidden_dim,
        },
    })

    if args.method == "simclr":
        cfg.method_kwargs.temperature = args.temperature
    elif args.method == "byol":
        cfg.method_kwargs.pred_hidden_dim = args.pred_hidden_dim
        cfg.momentum = omegaconf.OmegaConf.create({
            "base_tau": 0.99,
            "final_tau": 1.0,
            "classifier": False,
        })

    return cfg


def build_train_loader(args: argparse.Namespace) -> DataLoader:
    transform = FullTransformPipeline([NCropAugmentation(build_train_transform(args.dataset), 2)])  # type: ignore

    if args.dataset == "cifar100":
        train_dataset = prepare_datasets(
            dataset="cifar100",
            transform=transform,
            train_data_path=Path(args.data_dir),
            download=True,
        )
    else:
        train_dataset = dataset_with_index(ImageFolder)(  # type: ignore
            root=Path(args.data_dir) / "train",
            transform=transform,
        )

    return prepare_dataloader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)


def build_val_loader(args: argparse.Namespace) -> DataLoader:
    val_transform = build_val_transform(args.dataset)

    if args.dataset == "cifar100":
        _, val_dataset = prepare_cls_datasets(
            dataset="cifar100",
            T_train=transforms.ToTensor(),  # unused
            T_val=val_transform,
            train_data_path=Path(args.data_dir),
            val_data_path=Path(args.data_dir),
            download=True,
        )
    else:
        val_dataset = ImageFolder(Path(args.data_dir) / "val", transform=val_transform)

    return DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.gpus > 0,
    )


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    train_loader = build_train_loader(args)
    val_loader = build_val_loader(args)

    cfg = build_cfg(args)
    method_cls = SimCLR if args.method == "simclr" else BYOL
    model = method_cls(cfg)

    # Replace 7x7 stem with 3x3 — standard fix for small images (CIFAR / TinyImageNet)
    model.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.backbone.maxpool = nn.Identity()

    use_gpu = args.gpus > 0 and torch.cuda.is_available()
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=args.gpus if use_gpu else 1,
        accelerator="gpu" if use_gpu else "cpu",
        sync_batchnorm=use_gpu,
        precision=args.precision,
        check_val_every_n_epoch=args.val_every_n_epochs,
        callbacks=[EpochMetricsPrinter(log_params=vars(args), console=not args.no_console_log, openbayestool=args.openbayestool)],
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
