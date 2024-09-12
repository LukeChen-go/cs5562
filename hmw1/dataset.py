import torch
from torch.utils.data import Dataset

class AdvDataset(Dataset):
    def __init__(self, images, labels):
        super(AdvDataset, self).__init__()
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        images, labels = [], []
        for batch_images, batch_labels in batch:
            images.append(batch_images)
            labels.append(batch_labels)
        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=0)
        return {"image": images, "label": labels}
