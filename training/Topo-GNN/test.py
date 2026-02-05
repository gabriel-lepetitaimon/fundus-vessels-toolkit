from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToDevice

from fundus_vessels_toolkit.segment_to_graph.models.dataset import VBranchDigraphDataset
from fundus_vessels_toolkit.vascular_data_objects.vtree import VTree


def main():
    # mp.set_start_method("spawn", force=True)

    PATH = Path("/run/media/gaby/GREY SSD/PostDoc/DATA/Fundus/Fundus-AV")
    RAW = PATH / "1-images"
    AV = PATH / "2-av-pred_CLEM"
    TOPO = PATH / "3-topo"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset...")
    dataset = VBranchDigraphDataset.load_from_dirs(RAW, TOPO, av_dir=AV, resize_to=1024)

    dataloader = DataLoader(dataset, batch_size=8, num_workers=4)

    print("Iterating over dataloader...")
    for data in dataloader:
        continue


def main2():
    path = Path(
        "/tmp/fundus-vessels-toolkit/datasets-cache/c61375dc711eaff711e8ea27458ea99cdc85a8ad6212d73f48a8630d349befb0-1024"
    )
    path = path / "target-topo"

    for file in path.glob("*.npz"):
        tree = VTree.load(path / file)
        for curve in tree.geometric_data().branch_curve():
            assert len(curve) <= 500, f"Curve too long: {len(curve)}"


if __name__ == "__main__":
    main()
