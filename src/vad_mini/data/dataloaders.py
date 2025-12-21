# sec/vad_mini/data/dataloaders.py
import torch
from torch.utils.data import DataLoader


def collate_fn(batch):
    images = []
    labels = []
    defect_types = []
    masks = []
    
    for sample in batch:
        images.append(sample["image"])
        labels.append(sample["label"])
        defect_types.append(sample["defect_type"])
        masks.append(sample["mask"])
    
    batched_images = torch.stack(images, dim=0)
    batched_labels = torch.stack(labels, dim=0)
    batched_defect_types = defect_types
    
    if all(mask is None for mask in masks):
        batched_masks = None
    else:
        processed_masks = []
        for mask in masks:
            if mask is None:
                non_none_mask = next((m for m in masks if m is not None), None)
                if non_none_mask is not None:
                    zero_mask = torch.zeros_like(non_none_mask)
                else:
                    h, w = images[0].shape[1:]  # Assuming (C, H, W)
                    zero_mask = torch.zeros(h, w)
                processed_masks.append(zero_mask)
            else:
                processed_masks.append(mask)
        batched_masks = torch.stack(processed_masks, dim=0)
    
    return {
        "image": batched_images,
        "label": batched_labels,
        "defect_type": batched_defect_types,
        "mask": batched_masks
    }


def get_train_loader(dataset, batch_size=32, collate_fn=None, **kwargs):
    config = {
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": True,
    }
    config.update(kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        **config
    )


def get_test_loader(dataset, batch_size=32, collate_fn=None, **kwargs):
    config = {
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": True,
    }
    config.update(kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        **config
    )

def get_dataloader(dataset, batch_size=32, shuffle=False, num_workers=None, 
                   pin_memory=None, drop_last=False, collate_fn=None, **kwargs):
    config = {
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": True,
    }
    
    if num_workers is not None:
        config["num_workers"] = num_workers
    if pin_memory is not None:
        config["pin_memory"] = pin_memory
    
    config.update(kwargs)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn,
        **config
    )