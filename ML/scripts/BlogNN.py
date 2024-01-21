import json
import random
import numpy as np
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader
from .text_transforms import get_features

FEATURES_NUM = 770
SAMPLES_NUM = 459
TRAIN_SAMPLES = 380
VALID_SAMPLES = 60


class BlogClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_blog_classifier = nn.Sequential(
            nn.Linear(FEATURES_NUM, 16),
            nn.ELU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.base_blog_classifier(x)


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


def __save_model_weights(model_weights,
                         fp_to_save: str, name: str,
                         additional_formatting: str = ''):
    additional_formatting = '_' + additional_formatting if additional_formatting else ''
    fp = f"{fp_to_save}/{name}{additional_formatting}"
    torch.save(model_weights, f=fp)


def __load_model_weights(model, fp_to_load):
    model.load_state_dict(torch.load(fp_to_load))


def _test(model: nn.Module, test_dataloader: DataLoader,
          loss_fn, device,
          fp_to_load: str | None = None) -> float:
    if fp_to_load is not None:
        __load_model_weights(model, fp_to_load)
    model.eval()
    with torch.no_grad():
        test_loss, correct = 0.0, 0
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)

            prediction = model(X)
            test_loss += loss_fn(prediction[:, 1], y)
            correct += (torch.argmax(prediction, dim=1) == y).type(torch.int).sum().item()
        print(f"Test loss is {test_loss}")
        print(f"Correctness on test is {correct}")
    return test_loss


def _train(model: nn.Module, train_dataloader, device,
           loss_fn, optimizer: torch.optim.Optimizer,
           verbose: bool = False, valid_dataloader=None,
           n_epochs: int = 50,
           fp_to_save: str = '.', fp_to_load: str | None = None):
    no_profit_epoch_number = 0
    last_loss = 0.0

    model.train()
    for epoch in range(1, n_epochs + 1):
        if epoch % 10 == 0:
            print(f'Epoch {epoch} {"-" * 80}')
        for batch, (X, y) in enumerate(train_dataloader, start=1):
            X, y = X.to(device), y.to(device)

            prediction = model(X)
            loss = loss_fn(prediction[:, 1], y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if verbose and epoch % 10 == 0:
            print(f"Loss {loss.item()} on epoch {epoch}")
            if valid_dataloader:
                _test(model, valid_dataloader, loss_fn, device)
                model.train()

        if loss >= last_loss:
            no_profit_epoch_number += 1
            if no_profit_epoch_number > 5:
                __save_model_weights(model.state_dict(), '.',
                                     'blog_model', f"{epoch}")
                raise StopIteration(f"No profit. Last epoch: {epoch}, last loss: {loss}")
        else:
            no_profit_epoch_number = 0
        last_loss = loss

        if fp_to_load is not None and epoch % 10 == 0:
            __save_model_weights(model.state_dict(), fp_to_save,
                                 'blog_model', f"{epoch}")
    __save_model_weights(model.state_dict(), fp_to_save,
                         'last+blog_model')


def main():
    random.seed(42)
    np.random.seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Current computing device: {device}")

    neural_network = BlogClassifier().to(device)
    print(f"Model structure:")
    print(neural_network)

    with open('../JSONs/texts.json', 'r') as f:
        data = json.load(f)
    data = list(filter(lambda post: post['author'] and post['message'], data))

    texts = list(map(lambda el: el['message'], data))
    authors = map(lambda el: el['author'], data)

    texts_features = get_features(texts, mode='train')
    authors_labels = np.array(list(map(__define_label, authors)), dtype='float32')

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

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(neural_network.parameters(),
                                 lr=0.04,
                                 betas=(0.99, 0.999),
                                 eps=0.001)

    _train(neural_network, train_dataloader, device, loss_fn, optimizer, verbose=True,
           valid_dataloader=valid_dataloader, n_epochs=3000, fp_to_save='.')

    test_loss = _test(neural_network, test_dataloader, loss_fn, device,
                      fp_to_load='./blog_model')
    print(f"Loss on test dataset: {test_loss}")


if __name__ == '__main__':
    main()
