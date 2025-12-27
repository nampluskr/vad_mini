# experiments/test_trainer_stfpm.py
import os, sys
source_dir = os.path.join(os.path.dirname(__file__), "..", "src")
if source_dir not in sys.path:
    sys.path.insert(0, source_dir)

from vad_mini.utils import set_seed
from vad_mini.data.datasets import MVTecDataset
from vad_mini.data.dataloaders import get_train_loader, get_test_loader
from vad_mini.data.transforms import get_train_transform, get_test_transform, get_mask_transform


DATA_DIR = "/mnt/d/deep_learning/datasets/mvtec"
# DATA_DIR = "/home/namu/myspace/NAMU/datasets/mvtec"
CATEGORY = "bottle"
IMG_SIZE = 256
CROP_SIZE = None
BATCH_SIZE = 4
NORMALIZE = True
SEED = 42


if __name__ == "__main__":

    set_seed(SEED)

    #######################################################
    ## Load Datste and Dataloader
    #######################################################

    train_dataset = MVTecDataset(
        root_dir=DATA_DIR,
        category=CATEGORY,
        split="train",
        transform=get_train_transform(img_size=IMG_SIZE, crop_size=CROP_SIZE, normalize=NORMALIZE),
        mask_transform=get_mask_transform(img_size=IMG_SIZE if CROP_SIZE is None else CROP_SIZE),
    )
    test_dataset = MVTecDataset(
        root_dir=DATA_DIR,
        category=CATEGORY,
        split="test",
        transform=get_test_transform(img_size=IMG_SIZE, crop_size=CROP_SIZE, normalize=NORMALIZE),
        mask_transform=get_mask_transform(img_size=IMG_SIZE if CROP_SIZE is None else CROP_SIZE),
    )
    train_loader = get_train_loader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
        persistent_workers=False,
    )
    test_loader = get_test_loader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
        persistent_workers=False,
    )

    #######################################################
    ## Train Model
    #######################################################

    from vad_mini.models.cflow.trainer import CflowTrainer

    trainer = CflowTrainer(backbone="resnet18")
    trainer.fit(train_loader, max_epochs=10, valid_loader=test_loader)
    thresholds = trainer.calibrate_threshold(train_loader)



    
