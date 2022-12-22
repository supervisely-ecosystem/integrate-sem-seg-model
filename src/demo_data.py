import os
import supervisely as sly

demo_weights_url = "https://download.openmmlab.com/mmsegmentation/v0.5/poolformer/fpn_poolformer_s12_8x4_512x512_40k_ade20k/fpn_poolformer_s12_8x4_512x512_40k_ade20k_20220501_115154-b5aa2f49.pth"


def prepare_weights():
    if sly.is_production():
        pass
    else:
        weights_path = os.path.join(sly.env.folder(), "weights.pth")
        if sly.fs.file_exists(weights_path):
            print(f"Demo NN weights: {weights_path}")
        else:
            print("Downloading model weights, please wait a bit ...")
            sly.fs.download(demo_weights_url, weights_path)
            print(f"Demo NN weights downloaded: {weights_path}")
