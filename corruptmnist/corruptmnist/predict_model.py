import click
import torch

from corruptmnist.models.model import MyAwesomeModel


def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    return torch.cat([model(batch) for batch in dataloader], 0)


@click.command()
@click.argument("model_checkpoint")
@click.argument("model_input")
def cli(model_checkpoint, model_input):
    """Predict using a trained model checkpoint."""
    print(f"Predicting with model checkpoint {model_checkpoint} on input {model_input}")
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))
    input = torch.load(model_input)
    # input = MyDataset(input)
    input = torch.utils.data.DataLoader(input, batch_size=64, shuffle=True)
    output = predict(model, input)
    torch.save(output, r"reports/output/predictions.pt")

    pass


if __name__ == "__main__":
    cli()

    pass
