# cocobetter

Customized version of pycocotools. Should be a drop-in replacement for the official pycocotools, just with more features.

## What's different from the official?

* Provides custom max_dets up to 1000
* Support for calculating AP and AR at IoU threshold of 25% (official version does AP50, AP75, and AP)

## Wishlist / TODO:

* [ ] Pull in the faster eval code from detectron2 (if the license allows for it)
* [ ] Add PR curve generation
* [ ] Improve the .stats output to make it easier to pull out individual stats without using hardcoded ordinals/indexes
* [ ] Add per-class version of .stats (also make it a dict as described in previous bullet)
* [ ] Add tools and classes for manipulating coco formatted json
* [ ] Publish to pypi

## Installing (choose one method)

### Quick Install (Non-editable)

Use this method if you don't need to customize the code, and just want to install it into your python environment:

```bash
conda activate "YOUR_ENV_NAME"
python -m pip install \
    git+https://github.com/GiscardBiamby/cocobetter.git#egg=pycocotools\&subdirectory=PythonAPI
```

### Quick Install (Editable)

This will pull the repo into your local folder under `./src`, and install it into your python environment in develop mode. The conda package `pycocotools` will point to `./src` inside the folder you run this command from. Editing the code in `./src` it will immediately reflect in your python env (no need to reinstall).

```bash
conda activate "YOUR_ENV_NAME"
python -m pip install -e \
    git+https://github.com/GiscardBiamby/cocobetter.git#egg=pycocotools\&subdirectory=PythonAPI
```

OR, add this line to your `requirements.txt`:

```
-e git+https://github.com/GiscardBiamby/cocobetter.git#egg=pycocotools\&subdirectory=PythonAPI
```

### Method 3: Clone and install

```bash
conda activate "YOUR_ENV_NAME"
git clone git@github.com:GiscardBiamby/cocobetter.git
cd cocobetter
pip install -e .
```

## Usage

From your python project:

```python
from pycocotools.coco import coco
from pycocotools.cocoeval import COCOeval
```

## COCO API - <http://cocodataset.org/>

COCO is a large image dataset designed for object detection, segmentation, person keypoints detection, stuff segmentation, and caption generation. This package provides Matlab, Python, and Lua APIs that assists in loading, parsing, and visualizing the annotations in COCO. Please visit <http://cocodataset.org/> for more information on COCO, including for the data, paper, and tutorials. The exact format of the annotations is also described on the COCO website. The Matlab and Python APIs are complete, the Lua API provides only basic functionality.

In addition to this API, please download both the COCO images and annotations in order to run the demos and use the API. Both are available on the project website.
\-Please download, unzip, and place the images in: coco/images/
\-Please download and place the annotations in: coco/annotations/
For substantially more details on the API please see <http://cocodataset.org/#download>.

After downloading the images and annotations, run the Matlab, Python, or Lua demos for example usage.

To install:
\-For Python, run "make" under coco/PythonAPI
