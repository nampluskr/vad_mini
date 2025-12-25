# src/vad_mini/models/components/base_trainer.py

from abc import ABC, abstractmethod
from tqdm import tqdm

import torch
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torchmetrics.functional.classification import binary_roc


#####################################################################
# Early Stopper
#####################################################################

class EarlyStopper:
    def __init__(self, patience=10, min_delta=1e-3, mode="max", target_value=None, monitor="loss"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.target_value = target_value
        self.monitor = monitor
        self.reset()

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.target_reached = False

    def step(self, score):
        if self.target_value is not None:
            if self.mode == "max" and score >= self.target_value:
                self.target_reached = True
                self.early_stop = True
                return True
            if self.mode == "min" and score <= self.target_value:
                self.target_reached = True
                self.early_stop = True
                return True

        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        elif self.mode == "min":
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True

        return False

    def get_info(self):
        target_str = f", target={self.target_value}" if self.target_value is not None else ""
        if self.best_score is None:
            return f"Best {self.monitor}: N/A (patience: 0/{self.patience}{target_str})"
        return f"Best {self.monitor}: {self.best_score:.4f} (patience: {self.counter}/{self.patience}{target_str})"


#####################################################################
# Early Stopper
#####################################################################

class BaseTrainer(ABC):
    def __init__(self, model, loss_fn=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device) if isinstance(loss_fn, torch.nn.Module) else loss_fn
        self.optimizer = None
        self.scheduler = None

        self.global_epoch = 0
        self.num_epochs = None
        self.train_loader = None
        self.valid_loader = None

        self.train_early_stopper = None
        self.valid_early_stopper = None

        self.aucroc = BinaryAUROC().to(self.device)
        self.aupr = BinaryAveragePrecision().to(self.device)

    #######################################################
    # Abstract methods for each model
    #######################################################

    @abstractmethod
    def configure_optimizers(self):
        raise NotImplementedError

    @abstractmethod
    def configure_early_stoppers(self):
        raise NotImplementedError

    @abstractmethod
    def training_step(self, batch):
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch):
        raise NotImplementedError

    #######################################################
    # fit: train model for num_epochs or num_steps
    #######################################################

    def fit(self, train_loader, num_epochs, valid_loader=None):
        self.num_epochs = num_epochs
        self.num_steps = num_epochs * len(train_loader)
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.configure_optimizers()
        self.configure_early_stoppers()

        self.on_fit_start()
        self.on_train_start()

        for _ in range(num_epochs):
            self.on_train_epoch_start()
            train_outputs = self.train_one_epoch(train_loader)
            self.on_train_epoch_end(train_outputs)

            if valid_loader is not None:
                self.on_validation_epoch_start()
                valid_outputs = self.validate_one_epoch(valid_loader)
                self.on_validation_epoch_end(valid_outputs)

            if self.train_early_stop or self.valid_early_stop:
                break

        self.on_train_end()
        self.on_fit_end()

    #######################################################
    # Hooks
    #######################################################

    def on_fit_start(self): pass

    def on_train_start(self):
        self.early_stop_str = ""
        self.train_early_stop = False
        self.valid_early_stop = False
        self.current_epoch = 0
        print("\n*** Training start...")

    def on_train_epoch_start(self):
        self.global_epoch += 1
        self.current_epoch += 1

    def on_train_epoch_end(self, outputs):
        self.epoch_info = f"[{self.current_epoch:3d}/{self.num_epochs}]"
        self.train_info = ", ".join([f"{k}:{v:.3f}" for k, v in outputs.items()])
        if self.valid_loader is None:
            print(f"{self.epoch_info} {self.train_info}")

        if self.train_early_stopper is not None:
            metric_name = self.train_early_stopper.monitor
            self.train_early_stop = self.train_early_stopper.step(outputs[metric_name])

            if self.train_early_stopper.target_reached:
                self.early_stop_str += f"Training target readched! {self.train_early_stopper.get_info()}"
            elif self.train_early_stopper.early_stop:
                self.early_stop_str += f"Training Early Stopped! {self.train_early_stopper.get_info()}"

    def on_validation_epoch_start(self): pass

    def on_validation_epoch_end(self, outputs):
        valid_info = ", ".join([f"{k}:{v:.3f}" for k, v in outputs.items()])
        print(f"{self.epoch_info} {self.train_info} | (val) {valid_info}")

        if self.valid_early_stopper is not None:
            metric_name = self.valid_early_stopper.monitor
            self.valid_early_stop = self.valid_early_stopper.step(outputs[metric_name])

            if self.valid_early_stopper.target_reached:
                self.early_stop_str += f"Validation target readched! {self.valid_early_stopper.get_info()}"
            elif self.valid_early_stopper.early_stop:
                self.early_stop_str += f"Validation Early Stopped! {self.valid_early_stopper.get_info()}"

    def on_train_end(self):
        if self.train_early_stop or self.valid_early_stop:
            print(f">> {self.early_stop_str}")
        print("*** Training completed!")

    def on_fit_end(self): pass

    #######################################################
    # Train one epoch
    #######################################################

    @torch.enable_grad()
    def train_one_epoch(self, train_loader):
        self.model.train()
        outputs = {}
        num_images = 0

        with tqdm(train_loader, leave=False, ascii=True) as progress_bar:
            progress_bar.set_description(f">> Training")
            for batch in progress_bar:
                self.optimizer.zero_grad()
                batch_size = batch["image"].shape[0]
                num_images += batch_size

                batch_outputs = self.training_step(batch)
                loss = batch_outputs["loss"]

                loss.backward()
                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                for name, value in batch_outputs.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    outputs.setdefault(name, 0.0)
                    outputs[name] += value * batch_size

                progress_bar.set_postfix({name: f"{value / num_images:.3f}" for name, value in outputs.items()})

        return {name: value / num_images for name, value in outputs.items()}

    #######################################################
    # Validate one epoch
    #######################################################

    @torch.no_grad()
    def validate_one_epoch(self, valid_loader):
        self.model.eval()
        all_pred_scores = []
        all_labels = []

        self.aucroc.reset()
        self.aupr.reset()

        with tqdm(valid_loader, leave=False, ascii=True) as progress_bar:
            progress_bar.set_description(f">> Validation")
            for batch in progress_bar:
                predictions = self.validation_step(batch)
                pred_scores = predictions["pred_score"].flatten()
                labels = predictions["label"]

                all_pred_scores.append(pred_scores.cpu())
                all_labels.append(labels.cpu())

        all_pred_scores = torch.cat(all_pred_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        self.aucroc.update(all_pred_scores, all_labels)
        self.aupr.update(all_pred_scores, all_labels)

        fpr, tpr, thresholds = binary_roc(all_pred_scores, all_labels)
        j_scores = tpr - fpr

        return {
            "auroc": self.aucroc.compute().item(),
            "aupr": self.aupr.compute().item(),
            "threshold": thresholds[torch.argmax(j_scores)].item()
        }

    @torch.no_grad()
    def calibrate_threshold(self, dataloader):
        self.model.eval()
        all_scores = []

        for batch in dataloader:
            images = batch["image"].to(self.device)
            predictions = self.model(images)
            pred_scores = predictions["pred_score"].flatten()
            all_scores.append(pred_scores.cpu())

        all_scores = torch.cat(all_scores)

        thresholds = {}
        thresholds["99%"] = torch.quantile(all_scores, 0.99).item()
        thresholds["97%"] = torch.quantile(all_scores, 0.97).item()
        thresholds["95%"] = torch.quantile(all_scores, 0.95).item()
        thresholds["3-sigma"] = (all_scores.mean() + 3 * all_scores.std()).item()
        thresholds["2-sigma"] = (all_scores.mean() + 2 * all_scores.std()).item()
        thresholds["1-sigma"] = (all_scores.mean() + 1 * all_scores.std()).item()

        return thresholds