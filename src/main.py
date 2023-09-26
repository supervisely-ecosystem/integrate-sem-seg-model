import os
from typing_extensions import Literal
from typing import List, Any, Dict
import numpy as np
from dotenv import load_dotenv
import torch
import supervisely as sly

from mmcv import Config
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor
from mmseg.apis.inference import inference_segmentor, init_segmentor


load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

weights_url = "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b3_512x512_160k_ade20k/segformer_mit-b3_512x512_160k_ade20k_20210726_081410-962b98d2.pth"

class MyModel(sly.nn.inference.SemanticSegmentation):
    def load_on_device(
        self,
        model_dir: str,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        ####### CUSTOM CODE FOR MY MODEL STARTS (e.g. MMSEGMENTATION) #######
        weights_path = self.download(weights_url)
        cfg = Config.fromfile(os.path.join(model_dir, "model_config.py"))
        self.model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
        checkpoint = load_checkpoint(self.model, weights_path, map_location=device)
        self.class_names = checkpoint["meta"]["CLASSES"]
        self.model.CLASSES = self.class_names
        self.model.cfg = cfg
        self.model.to(device)
        self.model.eval()
        self.model = revert_sync_batchnorm(self.model)
        ####### CUSTOM CODE FOR MY MODEL ENDS (e.g. MMSEGMENTATION)  ########
        print(f"✅ Model has been successfully loaded on {device.upper()} device")

    def get_classes(self) -> List[str]:
        return self.class_names  # e.g. ["cat", "dog", ...]

    def predict(
        self, image_path: str, settings: Dict[str, Any]
    ) -> List[sly.nn.PredictionSegmentation]:

        ####### CUSTOM CODE FOR MY MODEL STARTS (e.g. DETECTRON2) #######
        segmented_image = inference_segmentor(self.model, image_path)[0]
        ####### CUSTOM CODE FOR MY MODEL ENDS (e.g. DETECTRON2)  ########

        return [sly.nn.PredictionSegmentation(segmented_image)]


model_dir = "my_model"  # model weights will be downloaded into this dir

m = MyModel(model_dir=model_dir)
m.load_on_device(model_dir=model_dir, device=device)

if sly.is_production():
    # this code block is running on Supervisely platform in production
    # just ignore it during development
    m.serve()
else:
    # for local development and debugging
    image_path = "./demo_data/image_01.jpg"
    results = m.predict(image_path, {})
    vis_path = "./demo_data/image_01_prediction.jpg"
    m.visualize(results, image_path, vis_path, thickness=2)
    print(f"predictions and visualization have been saved: {vis_path}")
