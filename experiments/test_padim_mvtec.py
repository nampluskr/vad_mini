# experiments/test_padim_mvtec.py

import os, sys
source_dir = os.path.join(os.path.dirname(__file__), "..", "src")
if source_dir not in sys.path:
    sys.path.insert(0, source_dir)

from defectvad.utils import set_seed
from defectvad.data.datasets import MVTecDataset
from defectvad.data.dataloaders import get_train_loader, get_test_loader
from defectvad.data.transforms import get_train_transform, get_test_transform, get_mask_transform


DATA_DIR = "/mnt/d/deep_learning/datasets/mvtec"
# DATA_DIR = "/home/namu/myspace/NAMU/datasets/mvtec"
CATEGORY = "capsule"
IMG_SIZE = 256
CROP_SIZE = None
CROP_SIZE = 224
BATCH_SIZE = 8
NORMALIZE = False
SEED = 42


if __name__ == "__main__":

    set_seed(SEED)

    #######################################################
    ## Load Dataset and Dataloader
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
    train_loader = get_train_loader(dataset=train_dataset, batch_size=BATCH_SIZE)
    test_loader = get_test_loader(dataset=test_dataset, batch_size=BATCH_SIZE)

    #######################################################
    ## Train Model
    #######################################################

    from defectvad.models.padim.trainer import PadimTrainer

    trainer = PadimTrainer(backbone="resnet18", layers=["layer1", "layer2", "layer3"])
    trainer.fit(train_loader, max_epochs=1)
    results = trainer.validate(test_loader)
    print(">> " + ", ".join([f"{k}:{v:.3f}" for k, v in results.items()]))

