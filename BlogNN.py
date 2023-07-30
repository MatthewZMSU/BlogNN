import json
import random
import numpy as np
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader
from text_transforms import get_features

SAMPLES_NUM = 459
TRAIN_SAMPLES = 380
VALID_SAMPLES = 60


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
        super().__init__()

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


def train(model: nn.Module, train_dataloader, device,
          loss_fn, optimizer: torch.optim.Optimizer,

          verbose: bool = False,
          n_epochs: int = 50):
    model.train()
    for epoch in range(1, n_epochs + 1):
        for batch, (X, y) in enumerate(train_dataloader, start=1):
            X, y = X.to(device), y.to(device)

            prediction = model(X)
            class_probabilities = torch.argmax(prediction, dim=1)
            loss = loss_fn(class_probabilities, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if verbose and batch % 10 == 0:
                print(f"Loss {loss.item()} on epoch {epoch}")


def main():
    random.seed(42)
    np.random.seed(42)

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
    authors_labels = np.array(list(map(__define_label, authors)), dtype=int)

    indices = np.arange(SAMPLES_NUM)
    np.random.shuffle(indices)
    train_indices = indices[:TRAIN_SAMPLES]
    valid_indices = indices[TRAIN_SAMPLES: TRAIN_SAMPLES + VALID_SAMPLES]
    test_indices = indices[TRAIN_SAMPLES + VALID_SAMPLES:]

    train_texts, train_labels = texts_features[train_indices], authors_labels[train_indices]
    valid_texts, valid_labels = texts_features[valid_indices], authors_labels[valid_indices]
    test_texts, test_labels = texts_features[test_indices], authors_labels[test_indices]

    train_dataset = BlogTextDataset(train_texts, train_labels)
    valid_dataset = BlogTextDataset(valid_texts, valid_labels)
    test_dataset = BlogTextDataset(test_texts, test_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(neural_network.parameters(),
                                lr=0.01, momentum=0.004)  # SGD with momentum

    train(neural_network, train_dataloader, device, loss_fn, optimizer, verbose=True)




if __name__ == '__main__':
    main()
