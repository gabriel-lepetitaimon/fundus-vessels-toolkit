import torch
import torch.utils.data as torch_data
import lightning as L
import torch_geometric as pyg


class GNNBranchTrainer(L.LightningModule):
    def __init__(self, model: torch.nn.Module, learning_rate: float = 1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss_fn(out, batch.y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer


class VBranchDigraphDataset(L.LightningDataModule):
    def prepare_data(self):
        datasets.VOCSegmentation(root="data", download=True)

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256, 256)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        target_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
        train_dataset = datasets.VOCSegmentation(root="data", transform=transform, target_transform=target_transform)
        return torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
