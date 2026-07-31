"""
Template for importing datasets for NN training.

Covers three tiers, roughly fastai-style ease -> raw control:
  1. torchvision.datasets   - vision benchmarks (MNIST, CIFAR10, FashionMNIST, etc.)
  2. huggingface `datasets` - huge hub of text/vision/audio datasets, one-liner download
  3. Custom Dataset class   - for your own files (images/csv/whatever)

Swap the ACTIVE section depending on what you need.
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def create_transform(mean: float = 0.5, std: float = 0.5) -> transforms.Compose:
    """Create a standard image preprocessing transform."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
        ]
    )


def load_torchvision_dataset(
    dataset_cls,
    root: str = "./data",
    train: bool = True,
    transform=None,
) -> Dataset:
    """Load a torchvision dataset purely."""
    return dataset_cls(root=root, train=train, download=True, transform=transform)


def create_data_loader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """Create a PyTorch DataLoader from a Dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def inspect_tensor_slice(
    tensor_data,
    row_slice=slice(4, 15),
    col_slice=slice(4, 22),
    font_size: str = "6pt",
):
    """Return a styled pandas DataFrame of a tensor slice for visual inspection."""
    if tensor_data.ndim == 3:
        tensor_data = tensor_data[0]

    df = pd.DataFrame(tensor_data[row_slice, col_slice].detach().cpu().numpy())
    return df.style.set_properties(**{"font-size": font_size}).background_gradient(
        "Greys"
    )


def filter_dataset_by_classes(dataset: Dataset, target_classes: tuple) -> Dataset:
    """Filter a dataset to keep only samples belonging to specific target classes."""
    indices = [i for i in range(len(dataset)) if int(dataset[i][1]) in target_classes]

    filtered_data = torch.stack([dataset[i][0] for i in indices])
    filtered_labels = torch.tensor([dataset[i][1] for i in indices])

    label_mapping = {label: new_idx for new_idx, label in enumerate(target_classes)}
    remapped_labels = torch.tensor([label_mapping[int(l)] for l in filtered_labels])

    return CustomDataset(filtered_data, remapped_labels)


# ---------------------------------------------------------------------------
# 3. Custom Dataset — for your own data (folder of images, csv, etc.)
# ---------------------------------------------------------------------------
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


# Example: ImageFolder for a directory laid out as root/class_x/xxx.png
# from torchvision.datasets import ImageFolder
# train_ds = ImageFolder(root="./data/train", transform=transform)

# ---------------------------------------------------------------------------
# quick sanity check & examples for train/test/validation loading
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    transform = create_transform()

    # 1. Torchvision: Load training and test/validation sets
    train_ds = load_torchvision_dataset(datasets.MNIST, train=True, transform=transform)
    test_ds = load_torchvision_dataset(datasets.MNIST, train=False, transform=transform)

    train_loader = create_data_loader(train_ds, batch_size=64, shuffle=True)
    test_loader = create_data_loader(test_ds, batch_size=64, shuffle=False)

    xb, yb = next(iter(train_loader))
    print(f"train batch shape: {xb.shape}, labels: {yb.shape}")

    xb_test, yb_test = next(iter(test_loader))
    print(f"test batch shape: {xb_test.shape}, labels: {yb_test.shape}")

    # Inspect a sample slice
    sample_img, sample_label = train_ds[0]
    print(f"Sample label: {sample_label}")
    # styled_df = inspect_tensor_slice(sample_img)

    # 1b. Filter dataset for specific classes (e.g., only 3s and 7s)
    train_3_7_ds = filter_dataset_by_classes(train_ds, target_classes=(3, 7))
    train_3_7_loader = create_data_loader(train_3_7_ds, batch_size=64, shuffle=True)
    xb_37, yb_37 = next(iter(train_3_7_loader))
    print(
        f"Filtered 3 & 7 batch shape: {xb_37.shape}, labels: {yb_37.unique().tolist()}"
    )

    # 2. Hugging Face Datasets example (text/vision/audio)
    # from datasets import load_dataset
    # hf_dataset = load_dataset("imdb")
    # hf_train_ds, hf_test_ds = hf_dataset["train"], hf_dataset["test"]

    # 3. Custom Dataset split example
    # split = int(len(data) * 0.8)
    # train_data, val_data = data[:split], data[split:]
    # train_labels, val_labels = labels[:split], labels[split:]
    #
    # custom_train_ds = CustomDataset(train_data, train_labels, transform=transform)
    # custom_val_ds = CustomDataset(val_data, val_labels, transform=transform)
    #
    # custom_train_loader = create_data_loader(custom_train_ds, batch_size=64, shuffle=True)
    # custom_val_loader = create_data_loader(custom_val_ds, batch_size=64, shuffle=False)
