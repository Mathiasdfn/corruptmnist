import matplotlib.pyplot as plt
import torch
from torch import nn, optim

from corruptmnist.data.make_dataset import MyDataset
from corruptmnist.models.model import MyAwesomeModel


def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel().to(device)
    train_set = MyDataset(
        (torch.load(r"data\processed\train_images.pt"), torch.load(r"data\processed\train_target.pt"))
    )
    train_set = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 30
    train_losses = []
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_set:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:
            print(f"Training loss: {running_loss/len(train_set)}")
            train_losses.append(running_loss / len(train_set))
    plt.plot(range(epochs), train_losses)
    plt.savefig(r"reports/figures/loss.png")
    torch.save(model.state_dict(), r"corruptmnist/models/checkpoints/checkpoint.pth")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(1e-3)
    pass
