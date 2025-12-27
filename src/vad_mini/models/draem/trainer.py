# src/vad_mini/models/draem/trainer.py

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.optim as optim

from vad_mini.common.base_trainer import BaseTrainer
from .torch_model import DraemModel
from .loss import DraemLoss
from vad_mini.components.perlin import PerlinAnomalyGenerator


class DraemTrainer(BaseTrainer):
    def __init__(self, dtd_dir, enable_sspcab=False, sspcab_lambda=0.1, beta=(0.1, 1.0)):

        model = DraemModel(sspcab=enable_sspcab)
        loss_fn = DraemLoss()
        super().__init__(model, loss_fn=loss_fn)

        self.augmenter = PerlinAnomalyGenerator(anomaly_source_path=dtd_dir, blend_factor=beta)
        self.sspcab = enable_sspcab

        if self.sspcab:
            self.sspcab_activations: dict = {}
            self.setup_sspcab()
            self.sspcab_loss = nn.MSELoss()
            self.sspcab_lambda = sspcab_lambda

    def setup_sspcab(self) -> None:
        def get_activation(name: str) -> Callable:
            def hook(_, __, output: torch.Tensor) -> None:  # noqa: ANN001
                self.sspcab_activations[name] = output
            return hook

        self.model.reconstructive_subnetwork.encoder.mp4.register_forward_hook(get_activation("input"))
        self.model.reconstructive_subnetwork.encoder.block5.register_forward_hook(get_activation("output"))

    def training_step(self, batch):
        input_image = batch["image"].to(self.device)
        augmented_image, anomaly_mask = self.augmenter(input_image)
        reconstruction, prediction = self.model(augmented_image)
        loss = self.loss_fn(input_image, reconstruction, anomaly_mask, prediction)

        if self.sspcab:
            loss += self.sspcab_lambda * self.sspcab_loss(
                self.sspcab_activations["input"],
                self.sspcab_activations["output"],
            )
        return {"loss": loss}

    def configure_optimizers(self):
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=0.0001)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[400, 600], gamma=0.1)
