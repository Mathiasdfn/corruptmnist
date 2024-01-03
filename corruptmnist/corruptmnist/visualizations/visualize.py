import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from corruptmnist.models.model import MyAwesomeModel

if __name__ == "__main__":
    # Load pre-trained network
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(r"corruptmnist/models/checkpoints/checkpoint.pth"))
    print(model.fc4)
    tsne = TSNE(n_components=2, perplexity=5)
    print(model.fc4.weight.data.numpy().shape)
    out = tsne.fit_transform(model.fc4.weight.data.numpy())
    plt.plot(out[:, 0], out[:, 1], "o")
    plt.savefig(r"reports/figures/tsne.png")

    pass
