# Fundus Vessels Toolkits

Fundus vessels toolkit is a collection of tools to analyze the vascular tree from fundus images.


## Installation


```bash
git clone https://github.com/gabriel-lepetitaimon/fundus-vessels-toolkit.git
pip install -e fundus-vessels-toolkit
```

## Vessels Segmentation and Classification

### Steered Convolutional Neuron
Steered CNN is a specific architecture to segment and classify vessels which implements rotational equivariance in CNN.
It is described in the - yet to be published - paper "Steered CNNs for Vessel Segmentation and Classification".

Steered Convolutionnal Neurons can be used as replacement of standard convolutional layer:
```python
from steered_cnn import SteeredConv2d, SteerableKernelBase

# Create a steerable kernel base equivalent to a 5x5 gaussian kernel (the actual kernel size is 7x7 to accommodate 45 degrees rotation).
steerable_base = SteerableKernelBase.create_radial(5)
steered_conv = SteeredConv2d(64, padding='same', steerable_base=steerable_base, nonlinearity='relu') 
```

or through predefined models architectures:
```python
from steered_cnn.models import SteeredUNet

model = SteeredUNet(3, 2, nfeatures=6, depth=2, nscale=5, base=steerable_base)
```

We provided pretrained models for some of these architectures. See the [pretrained models](#pretrained-models) section for more details.

### Metrics
Standard segmentation metrics are imperfect to measure the quality of vessels segmentation: besides the class imbalance,
they are often too sensitive to small variation on the main vessels boundaries while being too insensitive to miss 
detection of small vessels.

To overcome this issue, we propose to use the following metrics:
- **Skeleton Precision (SP)**: the percentage of pixels in the skeleton of the ground truth that were predicted as vessels.
- **Skeleton Recall (SR)**: the percentage of pixels in the skeleton of the prediction that were labelled as vessels.
- **Skeleton F1 (SF1)**: the harmonic mean of SP and SR.
- **Vascular Graph Distance (VGD)**:


### Pretrained Models
The models were pretrained on FIVES, DRIVE, MESSIDOR, IDRID (RETA) datasets.
