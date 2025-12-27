# src/vad_mini/models/fre/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim

from vad_mini.common.base_trainer import BaseTrainer
from .torch_model import FREModel


class FRETrainer(BaseTrainer):
    def __init__(self, backbone="resnet50", layer="layer3", pooling_kernel_size=2, 
        input_dim=65536, latent_dim=220):

        model = FREModel(
            backbone=backbone,
            layer=layer,
            pooling_kernel_size=pooling_kernel_size,
            input_dim=input_dim,
            latent_dim=latent_dim,
        )
        loss_fn = nn.MSELoss()
        super().__init__(model, loss_fn=loss_fn)

    def configure_optimizers(self):
        self.optimizer = optim.Adam(params=self.model.fre_model.parameters(), lr=1e-3)

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        features_in, features_out, _ = self.model.get_features(images)
        loss = self.loss_fn(features_in, features_out)
        return {"loss": loss}
