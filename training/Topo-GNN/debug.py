from pathlib import Path

from fundus_vessels_toolkit.segment_to_graph.models.dataset import VBranchDigraphDataset

PATH = Path("/run/media/gaby/GREY SSD/PostDoc/DATA/Fundus/Fundus-AV/")
RAW = PATH / "1-images"
GRAPH = PATH / "2-av-graph_CLEM"
TOPO = PATH / "3-topo"

dataset = VBranchDigraphDataset.load_from_dirs(RAW, TOPO, graph_dir=GRAPH)
print(len(dataset))
