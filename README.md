# Fundus Vessels Toolkits

Fundus vessels toolkit is a collection of tools to analyze the vascular tree from fundus images.


## Installation


```bash
git clone https://github.com/gabriel-lepetitaimon/fundus-vessels-toolkit.git
pip install -e fundus-vessels-toolkit
```

If you plan to use the commmon use-case example provided as Jupyter Notebooks in the `samples/` folder, you should 
also install their dependencies:
```bash
pip install -e "fundus-vessels-toolkit[samples]"
```

## Vessels Segmentation and Classification

### Pretrained Segmentation Models
For now, the only model pretrained model for vasculaire segmentation available through this repository 
is a UNet using a Resnet34 as a backbone and trained on FIVES.

But we aim to provide more models using different architecture and trained on more datasets in the near future.

```python
import cv2
from fundus_vessels_toolkit.models import segment

raw = cv2.imread('path/to/raw/fundus/image.jpg')
segmented = segment(raw)
```


### Steered Convolutional Neuron
Steered CNN is a specific architecture to segment and classify vessels which implements rotational equivariance in CNN.
It is described in the paper "Steered Convolutional Neurons for Retinal Vessels Classification" (ISBI2023). Please cite this paper if you want to use this code in a published work.

Steered Convolutionnal Neurons can be used as replacement of standard convolutional layer:
```python
from steered_cnn import SteeredConv2d, SteerableKernelBase

# Define batched image x and vector field α
x = torch.empty(8, 3, 512, 512)       # (b, [r,g,b], h, w)
α = torch.empty(8, 2, 512, 512 )  # (b, [cos(α),sin(α)], h, w) or simply torch.empty(8, 512, 512) to provide α in radians.


# Create a steerable kernel base equivalent to a 5x5 gaussian kernel (the actual kernel size is 7x7 to accommodate 45 degrees rotation).
steerable_base = SteerableKernelBase.create_radial(5)
steered_conv = SteeredConv2d(64, padding='same', steerable_base=steerable_base, nonlinearity='relu') 
y = steered_conv(x, α)
```

or through predefined models architectures:
```python
from steered_cnn.models import SteeredUNet

model = SteeredUNet(3, 2, nfeatures=6, depth=2, nscale=5, base=steerable_base)
pred = model(x, alpha=α)
```

