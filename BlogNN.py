import json
import random
import numpy as np
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader
from text_transforms import get_features


class BlogClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_blog_classifier = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.AvgPool1d(kernel_size=3, stride=3),  # (batch_size, 64)
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.layers(x)


class BlogTextDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def __define_label(author: str) -> int:
    """
    Maxim -> 0
    Matthew -> 1
    """
    if author == 'Максим Стрельцов':
        return 0
    elif author == 'Матвей Званцов':
        return 1
    else:
        raise ValueError(f"Not correct username from telegram: {author}")


def _split_data(data, indices):
    pass

def main():
    random.seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Current computing device: {device}")

    neural_network = BlogClassifier().to(device)
    print(f"Model structure:")
    print(neural_network)

    with open('texts.json', 'r') as f:
        data = json.load(f)
    data = list(filter(lambda post: post['author'] and post['message'], data))

    texts = list(map(lambda el: el['message'], data))
    authors = map(lambda el: el['author'], data)

    texts_features = get_features(texts, mode='train')
    authors_labels = np.array(map(__define_label, authors), dtype=int)

    dataset = BlogTextDataset(texts_features, )


if __name__ == '__main__':
    main()
