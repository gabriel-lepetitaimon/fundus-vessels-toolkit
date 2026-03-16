from datetime import datetime
from pathlib import Path

import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.loader import DataLoader as PyGDataLoader

import wandb
from fundus_vessels_toolkit.segment_to_graph.models.dataset import VBranchDigraphDataset
from fundus_vessels_toolkit.segment_to_graph.models.trainer import DigraphGNNTrainer

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.conv.fp32_precision = "tf32"
torch.backends.cuda.matmul.fp32_precision = "tf32"


def train():
    # Initialize a new W&B run (wandb.agent handles the config)
    wandb.init(project="GNN-topo-MICCAI")

    # Access the hyperparms assigned to this specific run
    config = wandb.config
    config.setdefaults(
        {
            "epoch": 160,
            "lr": 1e-2,
            # "weight_decay": 1e-5,
            # "batch_size": 4,
        }
    )

    # Setup the logger and trainer
    wandb_logger = WandbLogger(log_model=True)  # logs model checkpoints
    model = DigraphGNNTrainer(config)

    # === DATASET ===
    PATH = [
        Path("/run/media/gaby/GREY SSD/PostDoc/DATA/Fundus/" + folder)
        for folder in ["GAVE-train", "MAPLES-DR", "Fundus-AV", "LES-AV", "INSPIRE"]
    ]
    RAW = [path / "1-images" for path in PATH]
    AV = [path / "2-av-pred_CLEMENT" for path in PATH]
    TOPO = [path / "3-topo" for path in PATH]
    dataset = VBranchDigraphDataset.load_from_dirs(
        RAW,
        TOPO,
        av_dir=AV,
        resize_to=1024,
        root=str(Path(__file__).parent / "tmp/DATA2"),
        overwrite=False,
        # ignore_recent=datetime(2026, 2, 19),
        ignore_recent=datetime(2026, 3, 8),
    )
    train_set, val_set, test_set = dataset.split_loaders(train_ratio=0.7, val_ratio=0.15)
    train_loader = PyGDataLoader(train_set, batch_size=3, shuffle=True, num_workers=5)
    val_loader = PyGDataLoader(val_set, batch_size=6, num_workers=2)

    checkpoints = [ModelCheckpoint(monitor="val_tree-parent-acc", mode="max")]

    trainer = L.Trainer(
        max_epochs=config.epoch,
        logger=wandb_logger,
        enable_progress_bar=True,  # Optional: cleaner console output during sweeps
        check_val_every_n_epoch=20,
        accumulate_grad_batches=2,
        # gradient_clip_val=0.5,
        # gradient_clip_algorithm="value",wandb
        # num_sanity_val_steps=0,
        callbacks=checkpoints,
        precision="bf16-mixed",
    )

    trainer.fit(model, train_loader, val_loader)

    test_loader = PyGDataLoader(test_set, batch_size=6, num_workers=2)
    trainer.test(model, dataloaders=[test_loader], ckpt_path="best")

    # Finish the run
    wandb.finish()


if __name__ == "__main__":
    train()
