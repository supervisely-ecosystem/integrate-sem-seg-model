import os
import cv2
from dotenv import load_dotenv

import supervisely as sly

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog


load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

# Initialize Supervisely
api = sly.Api.from_env()
project_id = int(os.environ["modal.state.slyProjectId"])

project_meta_json = api.project.get_meta(project_id)
project_meta = sly.ProjectMeta.from_json(project_meta_json)


# Initialize Detectron2 model from config
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
predictor = DefaultPredictor(cfg)
class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get('thing_classes')


# Create meta from model classes and merge metas
model_obj_classes = []
for class_name in class_names:
    model_obj_classes.append(sly.ObjClass(class_name, sly.Bitmap))
confidence_tag_meta = sly.TagMeta(
    name='confidence', 
    value_type=sly.TagValueType.ANY_NUMBER, 
    applicable_to=sly.TagApplicableTo.OBJECTS_ONLY
)
model_meta = sly.ProjectMeta(
    obj_classes=model_obj_classes, 
    tag_metas=sly.TagMetaCollection([confidence_tag_meta])
)
merged_meta = project_meta.merge(model_meta)
api.project.update_meta(project_id, merged_meta)
# Question: do we want to change source project? maybe clone?


datasets = api.dataset.get_list(project_id)

# TODO: add progress
for dataset_info in datasets:
    ds_images = api.image.get_list(dataset_info.id)
    for image_info in ds_images:
        image = api.image.download_np(image_info.id)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_size = image.shape[:2]

        # Get predictions from Detectron2 model
        outputs = predictor(image)

        pred_classes = outputs["instances"].pred_classes.detach().numpy()
        pred_class_names = [class_names[pred_class] for pred_class in pred_classes]
        pred_scores = outputs["instances"].scores.detach().numpy()
        pred_masks = outputs["instances"].pred_masks.detach().numpy()

        # create Supervisely Labels from predicted objects
        labels = []
        for score, class_name, mask in zip(pred_scores, pred_class_names, pred_masks):
            if not mask.any(): # skip empty masks
                continue
            obj_class = merged_meta.get_obj_class(class_name)
            conf_tag = sly.Tag(
                meta=merged_meta.get_tag_meta('confidence'), 
                value=round(float(score), 4)
            )
            geometry = sly.Bitmap(mask)
            mask_label = sly.Label(geometry, obj_class, sly.TagCollection([conf_tag]))
            labels.append(mask_label)

        # create and upload Supervisely Annotation
        ann = sly.Annotation(img_size=img_size, labels=labels)
        api.annotation.upload_ann(image_info.id, ann)
