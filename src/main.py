from distutils.log import warn
import os
from typing import Literal, List
import cv2
import json
from dotenv import load_dotenv
import supervisely as sly
from typing_extensions import Literal

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

from supervisely.app.widgets import card

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

# code for detectron2 inference copied from official COLAB tutorial (inference section):
# https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
# https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html


class MyModel(sly.nn.inference.InstanceSegmentation):
    def load_on_device(
        self,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        ####### CODE FOR DETECTRON2 MODEL STARTS #######
        with open(os.path.join(model_dir, "model_info.json"), "r") as myfile:
            model_info = json.loads(myfile.read())
        cfg = get_cfg()
        cfg.merge_from_file(
            # Initialize Detectron2 model from config
            model_zoo.get_config_file(model_info["architecture"])
        )
        cfg.MODEL.DEVICE = device  # learn more in torch.device
        cfg.MODEL.WEIGHTS = os.path.join(model_dir, "model_weights.pkl")

        self.predictor = DefaultPredictor(cfg)
        self.class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get(
            "thing_classes"
        )
        ####### CODE FOR DETECTRON2 MODEL ENDS #########
        print(f"✅ Model has been successfully loaded on {device} device")

    def get_classes(self) -> list[str]:
        return self.class_names  # ["cat", "dog", ...]

    def predict(
        self, image_path: str, confidence_threshold: float = 0.8
    ) -> list[sly.nn.PredictionMask]:
        image = cv2.imread(image_path)  # BGR

        ####### CODE FOR DETECTRON2 MODEL STARTS #######
        outputs = self.predictor(image)  # get predictions from Detectron2 model
        pred_classes = outputs["instances"].pred_classes.detach().numpy()
        pred_class_names = [self.class_names[pred_class] for pred_class in pred_classes]
        pred_scores = outputs["instances"].scores.detach().numpy().tolist()
        pred_masks = outputs["instances"].pred_masks.detach().numpy()
        ####### CODE FOR DETECTRON2 MODEL ENDS #########

        results = []
        for score, class_name, mask in zip(pred_scores, pred_class_names, pred_masks):
            if score >= confidence_threshold:
                results.append(sly.nn.PredictionMask(class_name, mask, score))
        return results


team_id = int(os.environ["context.teamId"])
model_dir = os.path.abspath(os.environ["context.slyFolder"])
device = os.environ.get("modal.state.device", "cpu")  # @TODO: reimplement

m = MyModel(model_dir)

if sly.is_production():
    # code below is running on Supervisely platform in production
    # just ignore it during development and testing
    m.serve()
else:
    # for local development and debugging
    m.load_on_device("cpu")
    image_path = "./demo_data/image_01.jpg"
    confidence_threshold = 0.7
    results = m.predict(image_path, confidence_threshold)
    vis_path = "./demo_data/image_01_prediction.jpg"
    m.visualize(results, image_path, vis_path)
    print("predictions and visualization have been created")
