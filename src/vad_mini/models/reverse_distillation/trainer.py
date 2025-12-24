# src/vad_mini/models/reverse_distillaton/trainer.py

import torch
import torch.optim as optim

from vad_mini.models.components.base_trainer import BaseTrainer
from .torch_model import ReverseDistillationModel
from .loss import ReverseDistillationLoss
from .anomaly_map import AnomalyMapGenerationMode


class ReverseDistillationTrainer(BaseTrainer):
    def __init__(self, backbone="wide_resnet50_2", layers=["layer1", "layer2", "layer3"],
        input_size=(256, 256), anomaly_map_mode="add", pre_trained=True):

        model = ReverseDistillationModel(
            backbone=backbone,
            layers=layers,
            input_size=input_size,
            anomaly_map_mode=anomaly_map_mode,
            pre_trained=pre_trained,
        )
        loss_fn = ReverseDistillationLoss()
        super().__init__(model, loss_fn)

    def configure_optimizers(self):
        self.optimizer = optim.Adam(
            params=list(self.model.decoder.parameters()) + list(self.model.bottleneck.parameters()),
            lr=0.005,
            betas=(0.5, 0.99),
        )

    def configure_early_stoppers(self):
        self.train_early_stopper = None
        self.valid_early_stopper = None

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        loss = self.loss_fn(*self.model(images))
        return {"loss": loss}
    
    def validation_step(self, batch):
        images = batch["image"].to(self.device)
        predictions = self.model(images)
        return {**batch, **predictions}
    
