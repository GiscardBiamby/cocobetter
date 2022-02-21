
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, OrderedDict, Tuple


from ..coco import COCO

__all__ = ["CocoToDarknet"]


@dataclass
class bbox:
    """
    Data class to store a bounding box annotation instance
    """

    img_id: int
    cat_id: int
    x_center: float
    y_center: float
    width: float
    height: float


class Img:
    """A helper class to store image info and annotations."""

    anns: List[bbox]

    def __init__(self, id: int, filename: str, width: float, height: float) -> None:
        self.id: int = id
        self.filename: str = filename
        self.width: float = width
        self.height: float = height
        self.anns = []

    def add_ann(self, ann: bbox) -> None:
        """Add an annotation to the image"""
        self.anns.append(ann)

    def get_anns(self) -> List[bbox]:
        """
        Gets annotations, possibly filters them in prep for converting to yolo/Darknet
        format.
        """
        return self.anns

    def to_darknet(self, box: bbox) -> bbox:
        """Convert a BBox from coco to Darknet format"""
        # COCO bboxes define the topleft corner of the box, but yolo expects the x/y
        # coords to reference the center of the box. yolo also requires the coordinates
        # and widths to be scaled by image dims, down to the range [0.0, 1.0]
        return bbox(
            self.id,
            box.cat_id,
            (box.x_center + (box.width / 2.0)) / self.width,
            (box.y_center + (box.height / 2.0)) / self.height,
            box.width / self.width,
            box.height / self.height,
        )

    def write_darknet_anns(self, label_file) -> None:
        """Writes bounding boxes to specified file in yolo/Darknet format"""
        # It's a bit leaky abstraction to have Img handle writing to file but it's
        # convenient b/c we have access to img height and width here to scale the bbox
        # dims. Same goes for .to_darknet()
        anns = self.get_anns()
        for box in anns:
            box = self.to_darknet(box)
            label_file.write(
                f"{box.cat_id} {box.x_center} {box.y_center} {box.width} {box.height}\n"
            )

    def has_anns(self) -> List[bbox]:
        """
        Returns true if this image instance has at least one bounding box (after any
        filters are applied)
        """
        # TODO: Can add filter to only return true if annotations have non-zero area: I
        # saw around ~5 or 6 annotations in the v2_train_chipped.json that had zero
        # area, not sure if those might cause problems for yolo
        return self.anns

    def get_label_path(self, base_path: Path) -> Path:
        return base_path / self.filename.replace("jpeg", "txt").replace("jpg", "txt")

    def get_img_path(self, base_path: Path, dataset_name: str, data_split: str) -> Path:
        return base_path / dataset_name.replace("_tiny", "") / "images" / data_split / self.filename


class CocoToDarknet:
    """Class that helps convert an MS COCO formatted dataset to yolo/Darknet format"""

    @staticmethod
    def convert(ann_path: Path, base_path: Path, dataset_name: str, data_split: str) -> None:
        """Convert specified dataset to Darknet format.

        Details:
            - Labels are written to base_path/<dataset_name>/labels/<data_split>/*.txt
            - A file containing list of category names, is written to
                <base_path>/<dataset_name>.names
        """
        coco = COCO(ann_path)
        images = CocoToDarknet.build_db(coco)
        # Make paths:
        labels_path = base_path / dataset_name / "labels" / data_split
        labels_path.mkdir(parents=True, exist_ok=True)
        names_path = base_path / f"{dataset_name}.names"
        image_paths = CocoToDarknet.generate_label_files(
            images, labels_path, base_path, dataset_name, data_split
        )
        CocoToDarknet.generate_image_list(base_path, dataset_name, image_paths, data_split)
        CocoToDarknet.generate_names(names_path, coco)

    @staticmethod
    def generate_names(names_path: Path, coco: COCO) -> None:
        categories = [c["name"] + "\n" for c in coco.dataset["categories"]]
        with open(names_path, "w") as names_file:
            names_file.writelines(categories)

    @staticmethod
    def generate_label_files(
        images: Dict[int, Img],
        labels_path: Path,
        base_path: Path,
        dataset_name: str,
        data_split: str,
    ) -> List[str]:
        """
        Generates one .txt file for each image in the coco-formatted dataset. The .txt
        files contain the annotations in yolo/Darknet format.
        """
        # Convert:
        img_paths = set()
        for img_id, img in images.items():
            if img.has_anns():
                label_path = labels_path / img.get_label_path(labels_path)
                with open(label_path, "w") as label_file:
                    img.write_darknet_anns(label_file)
                img_path = img.get_img_path(base_path, dataset_name, data_split)
                assert img_path.exists(), f"Image doesn't exist {img_path}"
                img_paths.add(str(img_path) + "\n")
        return list(img_paths)

    @staticmethod
    def generate_image_list(
        base_path: Path, dataset_name: str, image_paths: List[str], data_split: str
    ) -> None:
        """Generates train.txt, val.txt, etc, txt file with list of image paths."""
        listing_path = base_path / dataset_name / f"{data_split}.txt"
        print("Listing path: ", listing_path)
        with open(listing_path, "w") as listing_file:
            listing_file.writelines(image_paths)

    @staticmethod
    def build_db(coco: COCO) -> Dict[int, Img]:
        """
        Builds a datastructure of images. All annotations are grouped into their
        corresponding images to facilitate generating the Darknet formatted metadata.

        Args:
            coco: a pycocotools.coco COCO instance

        Returns: Dictionary whose keys are image id's, and values are Img instances that
            are loaded with all the image info and annotations from the coco-formatted
            json
        """
        anns = coco.dataset["annotations"]
        images: Dict[int, Img] = {}
        # Build images data structure:
        for i, ann in enumerate(anns):
            ann = CocoToDarknet.get_ann(ann)
            if ann.img_id not in images:
                coco_img = coco.dataset["images"][ann.img_id]
                images[ann.img_id] = Img(
                    ann.img_id,
                    coco_img["file_name"],
                    float(coco_img["width"]),
                    float(coco_img["height"]),
                )
            img = images[ann.img_id]
            img.add_ann(ann)
        return images

    @staticmethod
    def get_ann(ann):
        """
        Gets a bbox instance from an annotation element pulled from the coco-formatted
        json
        """
        box = ann["bbox"]
        return bbox(ann["image_id"], ann["category_id"], box[0], box[1], box[2], box[3])
