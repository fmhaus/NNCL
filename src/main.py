import argparse
from pathlib import Path

import omegaconf
import lightning.pytorch as pl
from torchvision import transforms
from solo.methods import SimCLR
from solo.data.pretrain_dataloader import (
    NCropAugmentation,
    FullTransformPipeline,
    prepare_dataloader,
    prepare_datasets,
)
from torch.utils.data import DataLoader
from solo.data.classification_dataloader import prepare_datasets as prepare_cls_datasets
from logger import EpochMetricsPrinter


CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def build_cifar_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SimCLR pretraining on CIFAR-100")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--proj_output_dim", type=int, default=128)
    parser.add_argument("--proj_hidden_dim", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_every_n_epochs", type=int, default=1)
    return parser.parse_args()


def build_cfg(args: argparse.Namespace) -> omegaconf.DictConfig:
    cfg = omegaconf.OmegaConf.create({
        "method": "simclr",
        "backbone": {
            "name": "resnet18",
            "kwargs": {},
        },
        "data": {
            "dataset": "cifar100",
            "num_classes": 100,
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
            "temperature": args.temperature,
        },
    })
    return cfg


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    transform = FullTransformPipeline([NCropAugmentation(build_cifar_transform(), 2)]) # type: ignore

    train_dataset = prepare_datasets(
        dataset="cifar100",
        transform=transform,
        train_data_path=Path(args.data_dir),
        download=True,
    )
    train_loader = prepare_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    _, val_dataset = prepare_cls_datasets(
        dataset="cifar100",
        T_train=transforms.ToTensor(),  # unused, we only need val
        T_val=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
        ]),
        train_data_path=Path(args.data_dir),
        val_data_path=Path(args.data_dir),
        download=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.gpus > 0,
    )

    cfg = build_cfg(args)
    model = SimCLR(cfg)

    import torch
    use_gpu = args.gpus > 0 and torch.cuda.is_available()
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=args.gpus if use_gpu else 1,
        accelerator="gpu" if use_gpu else "cpu",
        sync_batchnorm=use_gpu,
        precision="16-mixed" if use_gpu else "32",
        check_val_every_n_epoch=args.val_every_n_epochs,
        callbacks=[EpochMetricsPrinter(log_params=vars(args))],
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
