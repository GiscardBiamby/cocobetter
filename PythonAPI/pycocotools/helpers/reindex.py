from ..coco import COCO

__all__ = ["ReIndex"]


class ReIndex(object):
    """
    A class used to reindex categories.

    Not sure if this class works or was ever used??
    """

    def __init__(self, coco: COCO):
        self.cats = coco.dataset["categories"]
        self.anns = coco.dataset["annotations"]
        self.id2name = {cat["id"]: cat["name"] for i, cat in enumerate(self.cats)}
        self.id2id = {cat["id"]: i + 1 for i, cat in enumerate(self.cats)}

        self.new_cats = [
            {
                "supercategory": cat["supercategory"],
                "id": self.id2id[cat["id"]],
                "name": cat["name"],
            }
            for cat in self.cats
        ]

        print("new cats: ", self.new_cats)

        self.new_anns = [
            {
                "segmentation": ann["segmentation"],
                "bbox": ann["bbox"],
                "area": ann["area"],
                "id": ann["id"],
                "image_id": ann["image_id"],
                "category_id": self.id2id[ann["category_id"]],
                "iscrowd": 0,  # matters for coco_eval
            }
            for ann in self.anns
        ]

    def coco_has_zero_as_background_id(self, coco):
        """
        Return true if category_id=0 is either unused, or used for background class. Else return
        false.
        """
        cat_id_zero_nonbackground_exists = False
        for cat in self.cats:
            if cat["id"] == 0:
                if cat["name"] not in ["background", "__background__"]:
                    cat_id_zero_nonbackground_exists = True
                    break
        # id:0 isn't used for any categories, so by default can assume it can be used for background
        # class:
        # if not cat_id_zero_nonbackground_exists:
        #     return True
        return not cat_id_zero_nonbackground_exists

        # # true if category_id=0 is either unused, or used for background class. Else return false.

        # if 0 not in list(self.id2id.keys()):
        #     self.cat_id_zero_nonbackground_exists = self.id2name[0] not in [
        #         "background",
        #         "__background__",
        #     ]
        # if cat["id"] == 0:
        #     if cat["name"] not in ["background", "__background__"]:
        #         cat_id_zero_nonbackground_exists = True
