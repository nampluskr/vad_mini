# experiments/load_stfpm.py
import os, sys
source_dir = os.path.join(os.path.dirname(__file__), "..", "src")
if source_dir not in sys.path:
    sys.path.insert(0, source_dir)

from vad_mini.models.dinomaly.torch_model import DinomalyModel


if __name__ == "__main__":

    model = DinomalyModel(
        encoder_name="dinov2reg_vit_base_14",
        bottleneck_dropout=0.2,
        decoder_depth=8,
        target_layers=None,
        fuse_layer_encoder=None,
        fuse_layer_decoder=None,
        remove_class_token=False,
    )