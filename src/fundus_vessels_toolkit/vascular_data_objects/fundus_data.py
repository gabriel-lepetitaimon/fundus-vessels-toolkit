from copy import copy
from enum import IntEnum
from pathlib import Path
from typing import Optional, Self, Tuple

import numpy as np
import numpy.typing as npt

from ..utils.data_io import load_image
from ..utils.geometric import Point
from ..utils.image import crop_pad_center
from ..utils.safe_import import is_torch_tensor

ABSENT = "ABSENT"


class AVLabel(IntEnum):
    """Enum class for the labels of the arteries and veins in the fundus image."""

    #: Background
    BKG = 0

    #: Artery
    ART = 1

    #: Vein
    VEI = 2

    #: Both
    BOTH = 3

    #: Unknown
    UNK = 4


class FundusData:
    def __init__(
        self,
        fundus=None,
        fundus_mask=None,
        vessels=None,
        od=None,
        od_center=None,
        macula=None,
        macula_center=None,
        *,
        name: Optional[str] = None,
        check: bool = True,
    ):
        shape = None
        if fundus is not None:
            self._fundus = self.load_fundus(fundus) if check else fundus
            shape = self._fundus.shape[:2]
        else:
            self._fundus = None

        if fundus_mask is not None:
            self._fundus_mask = self.load_fundus_mask(fundus_mask, shape) if check else fundus_mask
            shape = self._fundus_mask.shape[:2]
        elif self._fundus is not None:
            self._fundus_mask = self.load_fundus_mask(self._fundus)

        #
        if vessels is not None:
            self._vessels = self.load_vessels(vessels, shape) if check else vessels
            shape = self._vessels.shape[:2]
        else:
            self._vessels = None
        self._bin_vessels = None

        if od is not None:
            self._od, self._od_center = self.load_od_macula(od, shape) if check else (od, None)
            shape = self._od.shape[:2]
        else:
            self._od, self._od_center = None, None

        if macula is not None:
            self._macula, self._macula_center = self.load_od_macula(macula, shape) if check else (macula, None)
            shape = self._macula.shape[:2]
        else:
            self._macula, self._macula_center = None, None

        if shape is None:
            raise ValueError("No data was provided to initialize the FundusData.")
        self._shape = shape

        if name is None:
            if isinstance(fundus, (str, Path)):
                name = Path(fundus).stem
            elif isinstance(vessels, (str, Path)):
                name = Path(vessels).stem
            elif isinstance(od, (str, Path)):
                name = Path(od).stem
            elif isinstance(macula, (str, Path)):
                name = Path(macula).stem
        self._name = name

    def update(
        self,
        fundus=None,
        vessels=None,
        od=None,
        macula=None,
        name: Optional[str] = None,
    ) -> Self:
        other = copy(self)
        if fundus is not None:
            other._fundus = other.load_fundus(fundus)
        if vessels is not None:
            other._vessels = other.load_vessels(vessels, other.shape)
            other._bin_vessels = None
        if od is not None:
            other._od, other._od_center = other.load_od_macula(od, other.shape)
        if macula is not None:
            other._macula, other._macula_center = other.load_od_macula(macula, other.shape)
        if name is not None:
            other._name = name
        return other

    def remove_od_from_vessels(self):
        vessels = self._vessels.copy()
        vessels[self.od] = AVLabel.UNK
        return self.update(vessels=vessels)

    ####################################################################################################################
    #    === CHECK METHODS ===
    ####################################################################################################################
    @staticmethod
    def load_fundus(fundus) -> npt.NDArray[np.float32]:
        if isinstance(fundus, (str, Path)):
            fundus = load_image(fundus)
        elif is_torch_tensor(fundus):
            fundus = fundus.detach().cpu().numpy()

        assert isinstance(fundus, np.ndarray), "The image must be a numpy array."
        assert fundus.ndim == 3 and fundus.shape[2] == 3, "The image must be a color image."

        if fundus.dtype != np.float32:
            fundus = fundus.astype(np.float32)

        if not isinstance(fundus, np.ndarray):
            raise TypeError("The image must be a numpy array.")

        return fundus

    @staticmethod
    def load_fundus_mask(fundus_mask, shape: Optional[Tuple[int, int]] = None) -> npt.NDArray[np.bool_]:
        if isinstance(fundus_mask, (str, Path)):
            fundus_mask = load_image(fundus_mask, cast_to_float=False)
        elif is_torch_tensor(fundus_mask):
            fundus_mask = fundus_mask.detach().cpu().numpy()

        assert isinstance(fundus_mask, np.ndarray), "The image must be a numpy array."
        if fundus_mask.ndim == 2:
            if fundus_mask.dtype != bool:
                fundus_mask = fundus_mask > 127 if fundus_mask.dtype == np.uint8 else fundus_mask > 0.5
        elif fundus_mask.ndim == 3:
            r_hist = np.histogram(fundus_mask[:, :, 0].flatten(), bins=127, range=(0, 255))[0]
            if r_hist[0] + r_hist[-1] == np.prod(fundus_mask.shape[:2]):
                # If the image is binary
                return fundus_mask[:, :, 0] > 127
            else:
                # Assume the image is a fundus image
                from ..utils.fundus import fundus_ROI

                fundus_mask = fundus_ROI(fundus_mask)
        else:
            raise ValueError("Invalid fundus mask")

        if shape is not None and fundus_mask.shape != shape:
            H, W = shape
            h, w = fundus_mask.shape
            assert (
                abs(h - H) < H * 0.1 and abs(w - W) < W * 0.1
            ), "The vessels map doesn't have the same shape as the fundus image."
            fundus_mask = crop_pad_center(fundus_mask, shape)

        return fundus_mask

    @staticmethod
    def load_vessels(vessels, shape=None) -> npt.NDArray[np.uint8]:
        if isinstance(vessels, (str, Path)):
            vessels = load_image(vessels, cast_to_float=False)
        elif is_torch_tensor(vessels):
            vessels = vessels.detach().cpu().numpy()

        assert isinstance(vessels, np.ndarray), "The vessels map must be a numpy array."
        if vessels.ndim == 3:
            # Check if the image is a binary image
            MAX = 255 if vessels.dtype == np.uint8 else 1
            r_hist, _ = np.histogram(vessels[:, :, 0].flatten(), bins=127, range=(0, MAX))
            assert r_hist[0] + r_hist[-1] == np.prod(vessels.shape[:2]), "Vessels map should be binary image"
            vessels = vessels > MAX / 2

            if np.all(np.all(vessels, axis=2)):
                vessels = (vessels[:, :, 0] * AVLabel.UNK).astype(np.uint8)
            else:
                av_map = np.zeros(vessels.shape[:2], dtype=np.uint8)
                av_map[vessels[:, :, 0]] = AVLabel.ART
                av_map[vessels[:, :, 2]] = AVLabel.VEI
                av_map[vessels[:, :, 1]] = AVLabel.BOTH
                av_map[vessels[:, :, 2] & vessels[:, :, 0]] = AVLabel.BOTH
                vessels = av_map
        elif vessels.ndim == 2:
            if vessels.dtype == bool:
                vessels = (vessels * AVLabel.UNK).astype(np.uint8)
            else:
                # Image is either already a label map or only contains vessels segmentation
                if vessels.dtype == np.uint8:
                    if vessels.max() > AVLabel.UNK:
                        vessels = ((vessels > 127) * AVLabel.UNK).astype(np.uint8)
                elif np.issubdtype(vessels.dtype, np.integer):
                    assert vessels.min() >= 0 and vessels.max() <= AVLabel.UNK, "Invalid vessels map"
                    vessels = vessels.astype(np.uint8)
                else:
                    vessels = ((vessels > 0.5) * AVLabel.UNK).astype(np.uint8)

        if shape is not None and vessels.shape != shape:
            H, W = shape
            h, w = vessels.shape
            assert (
                abs(h - H) < H * 0.1 and abs(w - W) < W * 0.1
            ), "The vessels map doesn't have the same shape as the fundus image."
            vessels = crop_pad_center(vessels, shape)

        return vessels

    @staticmethod
    def load_od_macula(seg, shape=None) -> npt.NDArray[np.uint8]:
        from ..utils.safe_import import import_cv2 as cv2

        if isinstance(seg, (str, Path)):
            seg = load_image(seg, cast_to_float=False)
        elif is_torch_tensor(seg):
            seg = seg.detach().cpu().numpy()

        assert isinstance(seg, np.ndarray), "The optic disc map must be a numpy array."
        if seg.ndim == 3:
            seg = seg.mean(axis=2)

        assert seg.ndim == 2, "The optic disc map must be a grayscale image."
        if seg.dtype != bool:
            MAX = 255 if seg.dtype == np.uint8 else 1
            seg = seg > MAX / 2

        if shape is not None and seg.shape != shape:
            H, W = shape
            h, w = seg.shape
            assert (
                abs(h - H) < H * 0.1 and abs(w - W) < W * 0.1
            ), "The vessels map doesn't have the same shape as the fundus image."
            seg = crop_pad_center(seg, shape)

        _, labels, stats, centroids = cv2().connectedComponentsWithStats(seg.astype(np.uint8), 4, cv2().CV_32S)
        if stats.shape[0] == 1:
            return seg, ABSENT
        if stats.shape[0] == 2:
            return seg, Point(*centroids[1][::-1])
        largest_cc = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        return labels == largest_cc, Point(*centroids[1][::-1])

    ####################################################################################################################
    #    === PROPERTY ACCESSORS ===
    ####################################################################################################################
    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def has_name(self) -> bool:
        return self._name is not None

    @property
    def name(self) -> str:
        if self._name is None:
            raise AttributeError("The name of the fundus image was not provided.")
        return self._name

    @property
    def has_image(self) -> bool:
        return self._fundus is not None

    @property
    def image(self) -> npt.NDArray[np.float32]:
        """The fundus image in RGB format. Values are float32 in the range [0, 1]."""
        return self._fundus

    @property
    def has_vessels(self) -> bool:
        return self._vessels is not None

    @property
    def vessels(self) -> npt.NDArray[np.bool_]:
        """The binary segmentation of the vessels.

        Raises
        ------
        AttributeError
            If the vessels segmentation was not provided.
        """
        if self._vessels is None:
            raise AttributeError(
                "The vessels segmentation was not provided.\n"
                "See `fundus_vessels_toolkit.segment_vessels()` for automatic vessels segmentation models."
            )
        if self._bin_vessels is None:
            self._bin_vessels = self._vessels > 0
        return self._bin_vessels

    @property
    def av(self) -> npt.NDArray[np.uint8]:
        """The arteries and veins labels of the vessels segmentation. The labels are defined in the `AVLabel` enum.

        Raises
        ------
        AttributeError
            If the vessels segmentation was not provided.
        """
        if self._vessels is None:
            raise AttributeError("The arteries and veins segmentation was not provided.")
        return self._vessels

    @property
    def has_od(self) -> bool:
        return self._od is not None

    @property
    def od(self) -> npt.NDArray[np.float32]:
        """The binary segmentation of the optic disc.

        Raises
        ------
        AttributeError
            If the optic disc segmentation was not provided.
        """
        if self._od is None:
            raise AttributeError("The optic disc segmentation was not provided.")
        return self._od

    @property
    def od_center(self) -> Optional[Point]:
        """The center of the optic disc or None if the optic disc is not visible in this fundus.

        Raises
        ------
        AttributeError
            If the optic disc segmentation was not provided.
        """
        if self._od_center is None:
            if self._od is None:
                raise AttributeError("The optic disc segmentation was not provided.")
            else:
                _, self._od_center = self._check_od(self._od, self.shape)
        return None if self._od_center is ABSENT else self._od_center

    @property
    def has_macula(self) -> bool:
        return self._macula is not None

    @property
    def macula(self) -> npt.NDArray[np.float32]:
        """The binary segmentation of the macula.

        Raises
        ------
        AttributeError
            If the macula segmentation was not provided.
        """
        if self._macula is None:
            raise AttributeError("The macula segmentation was not provided.")
        return self._macula

    @property
    def macula_center(self) -> Optional[Point]:
        """The center of the macula or None if the macula is not visible in this fundus.

        Raises
        ------
        AttributeError
            If the macula segmentation was not provided.
        """
        if self._macula_center is None:
            if self._macula is None:
                raise AttributeError("The macula segmentation was not provided.")
            else:
                _, self._macula_center = self._check_od(self._macula, self.shape)
        return None if self._macula_center is ABSENT else self._macula_center

    @property
    def name(self) -> str:
        """The name of the fundus image.

        Raises
        ------
        AttributeError
            If the name of the fundus image was not provided.
        """
        if self._name is None:
            raise AttributeError("The name of the fundus image was not provided.")

        return self._name

    ####################################################################################################################
    #    === VISUALISATION TOOLS ===
    ####################################################################################################################
    def draw(self, labels_opacity=0.5, *, view=None):
        from jppype import Mosaic, View2D

        if isinstance(view, Mosaic):
            for v in view.views:
                self.draw(labels_opacity, view=v)
            return view
        elif view is None:
            view = View2D()
        view.add_image(self._fundus, "fundus")

        COLORS = {
            AVLabel.ART: "coral",
            AVLabel.VEI: "cornflowerblue",
            AVLabel.BOTH: "darkorchid",
            AVLabel.UNK: "gray",
            10: "white",
            11: "white",
        }

        labels = np.zeros(self.shape, dtype=np.uint8)
        if self._vessels is not None:
            labels = self._vessels

        if self._od is not None:
            labels[self._od] = 10

        if self._macula is not None:
            labels[self._macula] = 11

        view.add_label(labels, "AnatomicalData", opacity=labels_opacity, colormap=COLORS)
        return view

    def show(self, labels_alpha=0.5):
        self.draw(labels_alpha).show()
