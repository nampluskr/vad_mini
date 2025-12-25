# src/vad_mini/models/fastflow/trainer.py

import torch
import torch.optim as optim

from vad_mini.models.components.base_trainer import BaseTrainer
from .torch_model import FastflowModel
from .loss import FastflowLoss


class FastflowTrainer(BaseTrainer):
    def __init__(self, backbone="wide_resnet50_2", pre_trained=True, 
            flow_steps=8, conv3x3_only=False, hidden_ratio=1.0, input_size=(256, 256)):

        model = FastflowModel(
            input_size=input_size,
            backbone=backbone,
            pre_trained=pre_trained,
            flow_steps=flow_steps,
            conv3x3_only=conv3x3_only,
            hidden_ratio=hidden_ratio,
        )
        loss_fn = FastflowLoss()
        super().__init__(model, loss_fn=loss_fn)

    def configure_optimizers(self):
        self.optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=0.001,
            weight_decay=0.00001,
        )
        self.scheduler = None
        self.gradient_clip_val = None

    def configure_early_stoppers(self):
        self.train_early_stopper = None
        self.valid_early_stopper = None

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        hidden_variables, jacobians = self.model(images)
        loss = self.loss_fn(hidden_variables, jacobians)
        return {"loss": loss}

    def validation_step(self, batch):
        images = batch["image"].to(self.device)
        predictions = self.model(images)
        return {**batch, **predictions}
