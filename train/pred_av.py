from pathlib import Path

import tqdm

from fundus_toolkits import FundusData
from fundus_toolkits.utils.data_io import most_common_image_ext
from fundus_vessels_toolkit.models import segment_av


def predict_av():
    # === DATASET ===
    PATH = Path("/run/media/gaby/GREY SSD/PostDoc/DATA/Fundus/LES-AV")
    RAW = PATH / "1-images"
    OUT = PATH / "2-av-pred_CLEMENT"
    imgs = sorted(RAW.glob(f"*{most_common_image_ext(RAW)}"))
    print(f"Predicting AV for {len(imgs)} images in {RAW} and saving to {OUT}")
    for img in tqdm.tqdm(imgs, desc="Predicting AV"):
        fundus = FundusData(image=img)
        segment_av(fundus)
        fundus.write_image(av=OUT / (img.stem + ".png"))


if __name__ == "__main__":
    predict_av()
