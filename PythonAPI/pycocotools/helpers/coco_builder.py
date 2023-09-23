from copy import deepcopy
from pathlib import Path
from typing import Any, Union

from ..coco import COCO
from .utils import save_json

__all__ = ["CocoJsonBuilder", "COCOShrinker"]


class CocoJsonBuilder(object):
    """
    A class used to help build coco-formatted json from scratch.
    """

    def __init__(
        self,
        categories: dict[int, Any],
        subset_cat_ids: list[int] = [],
        dest_path: Union[str, Path] = "",
        dest_name="",
        keep_empty_images=True,
    ):
        """
        Args:

            categories: this can be the COCO.dataset['categories'] property if you
                are building a COCO json derived from an existing COCO json and don't want to modify
                the classes. It's a list of dictionary objects. Each dict has three keys: "id":int =
                category id, "supercatetory": str = name of parent category, and a "name": str =
                name of category.

            subset_cat_ids: list of category_id's. If specified, the builder will exclude
                annotations for any categories not in this list.

            dest_path: str or pathlib.Path instance, holding the path to directory where
                the new COCO formatted annotations file (dest_name) will be saved.

            dest_name: str of the filename where the generated json will be saved to.
        """
        if dest_path:
            if isinstance(dest_path, str):
                dest_path = Path(dest_path)
            assert dest_path.is_dir(), "dest_path should be a directory: " + str(
                dest_path
            )
        self.categories = categories
        self.subset_cat_ids = subset_cat_ids
        self.new_categories = []
        self.reindex_cat_id = {}  # maps from old to new cat id
        if self.subset_cat_ids:
            cat_counter = 1  # one-indexing
            for cat in self.categories:
                if cat["id"] in self.subset_cat_ids:
                    new_cat = deepcopy(cat)
                    new_cat["id"] = cat_counter
                    self.reindex_cat_id[cat["id"]] = cat_counter
                    cat_counter += 1
                    self.new_categories.append(new_cat)
        self.keep_empty_images = keep_empty_images
        self.dest_path = Path(dest_path)
        self.dest_name = dest_name
        self.images = []
        self.annotations: list[dict[str, Any]] = []
        dest_path.mkdir(parents=True, exist_ok=True)
        # assert self.dest_path.exists(), f"dest_path: '{self.dest_path}' does not exist"
        # assert (
        #     self.dest_path.is_dir()
        # ), f"dest_path: '{self.dest_path}' is not a directory"

    def generate_info(self) -> dict[str, str]:
        """
        Returns: A dictionary of descriptive info about the dataset.
        """
        info_json = {
            "description": "XView Dataset",
            "url": "http://xviewdataset.org/",
            "version": "1.0",
            "year": 2018,
            "contributor": "Defense Innovation Unit Experimental (DIUx)",
            "date_created": "2018/02/22",
        }
        return info_json

    def generate_licenses(self) -> list[dict[str, Any]]:
        """Returns the json hash for the licensing info."""
        return [
            {
                "url": "http://creativecommons.org/licenses/by-nc-sa/4.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
            }
        ]

    def add_image(self, img: dict[str, Any], annotations: list[dict]) -> None:
        """
        Add an image and it's annotations to the coco json.

        Args:
            img: A dictionary of image attributes. This gets added verbatim to the
                json, so in typical use cases when you are building a coco json from an existing
                coco json, you would just pull the entire coco.imgs[img_id] object and pass it as
                the value for this parameter.

            annotations: annotations of the image to add. list of dictionaries.
                Each dict is one annotation, it contains all the properties of the annotation that
                should appear in the coco json. For example, when using this json builder to build
                JSON's for a train/val split, the annotations can be copied straight from the coco
                object for the full dataset, and passed into this parameter.

        Returns: None
        """
        temp_anns = []
        for ann in annotations:
            # if builder was initialized with subset_cat_ids, only the corresponding annotations are
            # re-indexed and added
            if self.subset_cat_ids:
                if ann["category_id"] in self.subset_cat_ids:
                    new_ann = deepcopy(ann)
                    new_ann["category_id"] = self.reindex_cat_id[ann["category_id"]]
                    temp_anns.append(new_ann)
            else:
                temp_anns.append(ann)

        if self.subset_cat_ids:
            if temp_anns or self.keep_empty_images:
                self.images.append(img)
                for ann in temp_anns:
                    self.annotations.append(ann)
            else:
                pass  # no image added
        else:
            self.images.append(img)
            for ann in temp_anns:
                self.annotations.append(ann)

    def get_json(self) -> dict[str, object]:
        """Returns the full json for this instance of coco json builder."""
        root_json = {}
        if self.new_categories:
            root_json["categories"] = self.new_categories
        else:
            root_json["categories"] = self.categories
        root_json["info"] = self.generate_info()
        root_json["licenses"] = self.generate_licenses()
        root_json["images"] = self.images
        root_json["annotations"] = self.annotations
        return root_json

    def save(self) -> None:
        """Saves the json to the dest_path/dest_name location."""
        file_path = self.dest_path / self.dest_name
        dataset = self.get_json()
        print(
            f"Writing coco_builder (num_img: { len(dataset['images']) }, "
            f"num_ann: { len(dataset['annotations']) }) output to: '{file_path}'"
        )
        save_json(file_path, data=dataset)


class COCOShrinker:
    """
    Shrinker takes an MS COCO formatted dataset and creates a tiny version of it.
    """

    def __init__(self, dataset_path: Path, keep_empty_images=False) -> None:
        assert dataset_path.exists(), f"dataset_path: '{dataset_path}' does not exist"
        assert dataset_path.is_file(), f"dataset_path: '{dataset_path}' is not a file"
        self.base_path: Path = dataset_path.parent
        self.dataset_path: Path = dataset_path
        self.keep_empty_images = keep_empty_images

    def shrink(self, target_filename: str, size: int = 512) -> Path:
        """
        Create a toy sized version of dataset so we can use it just for testing if code
        runs, not for real training.

        Args:
            name: filename to save the tiny dataset to.
            size: number of items to put into the output. The first <size>
                elements from the input dataset are placed into the output.

        Returns: Path where the new COCO json file is saved.
        """
        # Create subset
        assert target_filename, "'target_filename' argument must not be empty"
        dest_path: Path = self.base_path / target_filename
        print(
            f"Creating subset of {self.dataset_path}, of size: {size}, at: {dest_path}"
        )
        coco = COCO(self.dataset_path)
        builder = CocoJsonBuilder(
            coco.dataset["categories"],
            dest_path=dest_path.parent,
            dest_name=dest_path.name,
        )
        subset_img_ids = coco.getImgIds()[:size]
        for img_id in subset_img_ids:
            anns = coco.imgToAnns[img_id]
            if anns or self.keep_empty_images:
                builder.add_image(coco.imgs[img_id], anns)
        builder.save()
        return dest_path
