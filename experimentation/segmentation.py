from nntemplate.callbacks.log_artifacts import Export2DLabel
from nntemplate.cli import cli


cmap_av = {(0, 0): 'blue', (1, 1): 'red', (1, 0): 'cyan', (0, 1): 'pink', 'default': 'lightgray'}
cmap_vessel = {(0, 0): '#edf6f9', (1, 1): '#83c5be', (1, 0): '#e29578', (0, 1): '#006d77', 'default': 'lightgray'}

Export2DLabel(cmap_vessel, every_n_epoch=10)
Export2DLabel(cmap_vessel, dataset_names=net.test_dataloaders_names)