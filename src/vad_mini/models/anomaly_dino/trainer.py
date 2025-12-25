# src/vad_mini/models/anomaly_dino/trainer.py

import torch
import torch.optim as optim

from vad_mini.models.components.base_trainer import BaseTrainer
from .torch_model import AnomalyDINOModel
from .loss import STFPMLoss


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

    def configure_optimizers(self):
        self.optimizer = optim.SGD(
            params=self.model.student_model.parameters(),
            lr=0.4,         # default lr=0.4
            momentum=0.9,
            dampening=0.0,
            weight_decay=0.001,
        )
        self.scheduler = None
        self.gradient_clip_val = 1.0

    def configure_early_stoppers(self):
        self.train_early_stopper = None
        self.valid_early_stopper = None

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        teacher_features, student_features = self.model.forward(images)
        loss = self.loss_fn(teacher_features, student_features)
        return dict(loss=loss)

    def validation_step(self, batch):
        images = batch["image"].to(self.device)
        predictions = self.model(images)
        return {**batch, **predictions}
