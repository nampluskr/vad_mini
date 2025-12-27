# src/vad_mini/models/anomaly_dino/trainer.py

import torch
import torch.optim as optim

from vad_mini.common.base_trainer import BaseTrainer
from .torch_model import AnomalyDINOModel


class AnomalyDINOTrainer(BaseTrainer):
    def __init__(self, encoder_name="dinov2_vit_small_14", num_neighbours=1,
        masking=False, coreset_subsampling=False, sampling_ratio=0.1):

        model = AnomalyDINOModel(
            num_neighbours=num_neighbours,
            encoder_name=encoder_name,
            masking=masking,
            coreset_subsampling=coreset_subsampling,
            sampling_ratio=sampling_ratio,
        )
        super().__init__(model, loss_fn=None)

    def on_train_start(self):
        super().on_train_start()
        self.max_epochs = 1

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        _ = self.model(images)
        return {"loss": torch.tensor(0.0).float().to(self.device)}

    def on_train_end(self):
        super().on_train_end()
        self.model.fit()