# src/defectvad/models/patchcore/trainer.py

import torch
import torch.optim as optim

from defectvad.common.base_trainer import BaseTrainer
from .torch_model import PatchcoreModel


class PatchcoreTrainer(BaseTrainer):
    def __init__(self, backbone="wide_resnet50_2", layers=["layer2", "layer3"],
        num_neighbors=9, coreset_sampling_ratio=0.1):

        model = PatchcoreModel(
            backbone=backbone,
            pre_trained=True,
            layers=layers,
            num_neighbors=num_neighbors,
        )
        super().__init__(model, loss_fn=None)

        self.coreset_sampling_ratio = coreset_sampling_ratio

    def on_train_start(self):
        super().on_train_start()
        self.max_epochs = 1

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        _ = self.model(images)
        return {"loss": torch.tensor(0.0).float().to(self.device)}

    def on_train_end(self):
        super().on_train_end()
        self.model.subsample_embedding(sampling_ratio=self.coreset_sampling_ratio)
