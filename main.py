import click
import torch
from model import MyAwesomeModel
from torch import nn, optim
import matplotlib.pyplot as plt

from data import mnist

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel().to(device)
    train_set, _ = mnist()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    epochs = 30
    steps = 0
    train_losses = []
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_set:
            images, labels = images.to(device), labels.to(device)
            # print(images.shape, labels.shape)
            # labels = labels.squeeze()
            optimizer.zero_grad()
            
            log_ps = model(images)
            # print(log_ps.shape)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            
        else:
            print(f"Training loss: {running_loss/len(train_set)}")
            train_losses.append(running_loss/len(train_set))
    plt.plot(range(epochs), train_losses)
    plt.savefig('loss.png')
    torch.save(model.state_dict(), 'checkpoint.pth')




@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)
    if not model_checkpoint:
        model_checkpoint = 'checkpoint.pth'
    # TODO: Implement evaluation logic here
    model = MyAwesomeModel().to(device)
    model.load_state_dict(torch.load(model_checkpoint))
    criterion = nn.NLLLoss()
    _, testloader = mnist()
    with torch.no_grad():
        sumaccuracy = 0
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            log_ps = model(images)
            test_loss = criterion(log_ps, labels)
            
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            sumaccuracy += accuracy.item()

        sumaccuracy /= len(testloader)
        print(f'Accuracy: {sumaccuracy*100}%')


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
