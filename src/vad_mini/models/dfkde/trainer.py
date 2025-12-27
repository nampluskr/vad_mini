# src/vad_mini/models/dfkde/trainer.py

import torch
import torch.optim as optim

from vad_mini.common.base_trainer import BaseTrainer
from .torch_model import DfkdeModel


class DfkdeTrainer(BaseTrainer):
    def __init__(self, backbone="resent18", layers=["layer4"], 
        n_pca_components=16, feature_scaling_method="scale", max_training_points=40000):

        model = DfkdeModel(
            layers=layers,
            backbone=backbone,
            n_pca_components=n_pca_components,
            feature_scaling_method=feature_scaling_method,
            max_training_points=max_training_points,
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
