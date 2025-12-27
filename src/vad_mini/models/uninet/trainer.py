# src/vad_mini/models/uninet/trainer.py

import torch
import torch.optim as optim

from vad_mini.common.base_trainer import BaseTrainer
from .torch_model import UniNetModel
from .loss import UniNetLoss


class UniNetTrainer(BaseTrainer):
    def __init__(self, student_backbone="wide_resnet50_2", teacher_backbone="wide_resnet50_2", temperature=0.1):

        model = UniNetModel(
            student_backbone=student_backbone, 
            teacher_backbone=teacher_backbone, 
            loss=UniNetLoss(temperature=temperature)
        )
        super().__init__(model, loss_fn=None)

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(
            [
                {"params": self.model.student.parameters()},
                {"params": self.model.bottleneck.parameters()},
                {"params": self.model.dfs.parameters()},
                {"params": self.model.teachers.target_teacher.parameters(), "lr": 1e-6},
            ],
            lr=5e-3,
            betas=(0.9, 0.999),
            weight_decay=1e-5,
            eps=1e-10,
            amsgrad=True,
        )
        milestones = [int(self.max_steps * 0.8) if self.max_steps != -1 else (self.trainer.max_epochs * 0.8)]
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.2)

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        masks = None
        # labels = None
        # masks = batch["mask"].to(self.device)
        labels = batch["label"].to(self.device)
        loss = self.model(images=images, masks=masks, labels=labels)
        return {"loss": loss}
