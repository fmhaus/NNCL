import argparse
from pathlib import Path

import pytorch_lightning as pl
from solo.methods import SimCLR
from solo.utils.pretrain_dataloader import (
    CifarTransform,
    prepare_dataloader,
    prepare_datasets,
    prepare_n_crop_transform,
)


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
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    # Two augmented views per image (standard SimCLR)
    cifar_transform = CifarTransform(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1,
        color_jitter_prob=0.8,
        gray_scale_prob=0.2,
        horizontal_flip_prob=0.5,
        gaussian_prob=0.0,
        solarization_prob=0.0,
        crop_size=32,
    )
    transform = prepare_n_crop_transform([cifar_transform], num_crops_per_aug=[2])

    train_dataset = prepare_datasets(
        dataset="cifar100",
        transform=transform,
        data_dir=Path(args.data_dir),
        download=True,
    )
    train_loader = prepare_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = SimCLR(
        # SimCLR
        proj_output_dim=args.proj_output_dim,
        proj_hidden_dim=args.proj_hidden_dim,
        temperature=args.temperature,
        supervised=False,
        # Backbone
        encoder="resnet18",
        num_classes=100,
        backbone_args={"cifar": True},
        # Training
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        optimizer="sgd",
        lars=True,
        lr=args.lr,
        weight_decay=args.weight_decay,
        classifier_lr=0.1,
        exclude_bias_n_norm=True,
        accumulate_grad_batches=1,
        extra_optimizer_args={"momentum": 0.9},
        scheduler="warmup_cosine",
        min_lr=0.0,
        warmup_start_lr=0.003,
        warmup_epochs=10,
        num_large_crops=2,
        num_small_crops=0,
        eta_lars=1e-3,
        grad_clip_lars=False,
        knn_eval=True,
        knn_k=20,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        sync_batchnorm=True,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        precision=16,
    )
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
