# src/vad_mini/models/cfa/trainer.py

from tqdm import tqdm
import torch
import torch.optim as optim

from vad_mini.common.base_trainer import BaseTrainer
from .torch_model import CfaModel
from .loss import CfaLoss


class CfaTrainer(BaseTrainer):
    def __init__(self, backbone="wide_resnet50_2", gamma_c=1, gamma_d=2,
        num_nearest_neighbors=3, num_hard_negative_features=3, radius=1e-5):

        model = CfaModel(
            backbone=backbone,
            gamma_c=gamma_c,
            gamma_d=gamma_d,
            num_nearest_neighbors=num_nearest_neighbors,
            num_hard_negative_features=num_hard_negative_features,
            radius=radius,
        )
        loss_fn = CfaLoss(
            num_nearest_neighbors=num_nearest_neighbors,
            num_hard_negative_features=num_hard_negative_features,
            radius=radius,
        )
        super().__init__(model, loss_fn=loss_fn)

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(
            params=self.model.parameters(),
            lr=1e-3,
            weight_decay=5e-4,
            amsgrad=True,
        )

    def backward(self, loss):
        loss.backward(retain_graph=True)

    def on_train_start(self):
        super().on_train_start()
        self.model.initialize_centroid(data_loader=self.train_loader)

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        distance = self.model(images)
        loss = self.loss_fn(distance)
        return {"loss": loss}
