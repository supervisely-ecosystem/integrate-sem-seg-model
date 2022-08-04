import os
import cv2
from tqdm import tqdm
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
src_project_info = api.project.get_info_by_id(project_id)
workspace_id = src_project_info.workspace_id
src_project_name = src_project_info.name
dst_project_name = f"{src_project_name}_labeled_inst_seg"

# Clone source project without existing annotations
clone_task_id = api.project.clone_advanced(
    project_id, workspace_id, dst_project_name, with_annotations=False
)
api.task.wait(clone_task_id, api.task.Status("finished"))
dst_project_info = api.project.get_info_by_name(workspace_id, dst_project_name)

# Get new project meta
project_meta_json = api.project.get_meta(dst_project_info.id)
project_meta = sly.ProjectMeta.from_json(project_meta_json)


def load_model():
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
    class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes")
    return predictor, class_names


def get_model_classes(class_names):
    # Create meta from model classes and merge metas
    model_obj_classes = []
    for class_name in class_names:
        model_obj_classes.append(sly.ObjClass(class_name, sly.Bitmap))
    confidence_tag_meta = sly.TagMeta(
        name="confidence",
        value_type=sly.TagValueType.ANY_NUMBER,
        applicable_to=sly.TagApplicableTo.OBJECTS_ONLY,
    )
    model_meta = sly.ProjectMeta(
        obj_classes=model_obj_classes, tag_metas=[confidence_tag_meta]
    )
    merged_meta = project_meta.merge(model_meta)
    api.project.update_meta(dst_project_info.id, merged_meta)
    return merged_meta


def apply_model_to_image(image, model, class_names, project_meta):
    img_size = image.shape[:2]
    # Get predictions from Detectron2 model
    outputs = model(image)

    pred_classes = outputs["instances"].pred_classes.detach().numpy()
    pred_class_names = [class_names[pred_class] for pred_class in pred_classes]
    pred_scores = outputs["instances"].scores.detach().numpy()
    pred_masks = outputs["instances"].pred_masks.detach().numpy()

    # create Supervisely Labels from predicted objects
    labels = []
    for score, class_name, mask in zip(pred_scores, pred_class_names, pred_masks):
        if not mask.any():  # skip empty masks
            continue
        obj_class = project_meta.get_obj_class(class_name)
        conf_tag = sly.Tag(
            meta=project_meta.get_tag_meta("confidence"),
            value=round(float(score), 4),
        )
        geometry = sly.Bitmap(mask)
        mask_label = sly.Label(geometry, obj_class, sly.TagCollection([conf_tag]))
        labels.append(mask_label)

    # create Supervisely Annotation
    annotation = sly.Annotation(img_size=img_size, labels=labels)
    return annotation


def main():
    model, class_names = load_model()
    merged_project_meta = get_model_classes(class_names)

    datasets = api.dataset.get_list(dst_project_info.id)
    imgs_num = sum([dataset.images_count for dataset in datasets])
    with tqdm(total=imgs_num) as pbar:
        for dataset_info in datasets:
            ds_images = api.image.get_list(dataset_info.id)
            for image_info in ds_images:
                image = api.image.download_np(image_info.id)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                annotation = apply_model_to_image(
                    image, model, class_names, merged_project_meta
                )
                api.annotation.upload_ann(image_info.id, annotation)
                pbar.update()


if __name__ == "__main__":
    main()
