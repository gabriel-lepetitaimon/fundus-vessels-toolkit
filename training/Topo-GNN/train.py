from pathlib import Path

import pytorch_lightning as L
import torch
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
    wandb.init(project="GNN-topo-debug")

    # Access the hyperparms assigned to this specific run
    config = wandb.config

    # Setup the logger and trainer
    wandb_logger = WandbLogger(log_model="all")  # logs model checkpoints
    model = DigraphGNNTrainer(config)

    # === DATASET ===
    PATH = [
        Path("/run/media/gaby/GREY SSD/PostDoc/DATA/Fundus/" + folder)
        for folder in ["GAVE-train", "MAPLES-DR", "Fundus-AV"]
    ]
    RAW = [path / "1-images" for path in PATH]
    AV = [path / "2-av-pred_CLEMENT" for path in PATH]
    TOPO = [path / "3-topo" for path in PATH]
    dataset = VBranchDigraphDataset.load_from_dirs(RAW, TOPO, av_dir=AV, resize_to=1024)
    train_set, val_set, test_set = dataset.split_loaders(train_ratio=0.75, val_ratio=0.15)
    train_loader = PyGDataLoader(train_set, batch_size=4, shuffle=True, num_workers=5)
    val_loader = PyGDataLoader(val_set, batch_size=2, num_workers=5)

    trainer = L.Trainer(
        max_epochs=config.get("epoch", 100),
        logger=wandb_logger,
        enable_progress_bar=True,  # Optional: cleaner console output during sweeps
        check_val_every_n_epoch=10,
        accumulate_grad_batches=2,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="value",
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_loader, val_loader)

    test_loader = PyGDataLoader(test_set, batch_size=1, num_workers=1)
    trainer.test(model, dataloaders=[test_loader])

    # Finish the run
    wandb.finish()


if __name__ == "__main__":
    train()
