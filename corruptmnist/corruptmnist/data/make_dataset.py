import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, data):
        self.images, self.labels = data

    def __getitem__(self, index):
        # Modify this to return a tuple of (image, label)
        return (self.images[index], self.labels[index])

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    # Get the data and process it
    trainimgs = torch.stack(
        (
            torch.load(r"data\raw\train_images_0.pt"),
            torch.load(r"data\raw\train_images_1.pt"),
            torch.load(r"data\raw\train_images_2.pt"),
            torch.load(r"data\raw\train_images_3.pt"),
            torch.load(r"data\raw\train_images_4.pt"),
            torch.load(r"data\raw\train_images_5.pt"),
        ),
        dim=1,
    )
    trainimgs = trainimgs.view(-1, 1, 28, 28)
    trainimgs = trainimgs.squeeze()
    trainlabels = torch.stack(
        (
            torch.load(r"data\raw\train_target_0.pt"),
            torch.load(r"data\raw\train_target_1.pt"),
            torch.load(r"data\raw\train_target_2.pt"),
            torch.load(r"data\raw\train_target_3.pt"),
            torch.load(r"data\raw\train_target_4.pt"),
            torch.load(r"data\raw\train_target_5.pt"),
        ),
        dim=1,
    )
    trainlabels = trainlabels.view(-1, 1)
    trainlabels = trainlabels.squeeze()
    trainimgs = transforms.Normalize(0, 1)(trainimgs)
    # train = MyDataset((trainimgs, trainlabels))
    # train = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    testimgs = torch.load(r"data\raw\test_images.pt")
    testlabels = torch.load(r"data\raw\test_target.pt")
    testimgs = testimgs.view(-1, 1, 28, 28)
    testimgs = testimgs.squeeze()
    testlabels = testlabels.view(-1, 1)
    testlabels = testlabels.squeeze()
    testimgs = transforms.Normalize(0, 1)(testimgs)
    # test = MyDataset((testimgs, testlabels))
    # test = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)
    torch.save(trainimgs, r"./data/processed/train_images.pt")
    torch.save(trainlabels, r"./data/processed/train_target.pt")
    torch.save(testimgs, r"./data/processed/test_images.pt")
    torch.save(testlabels, r"./data/processed/test_target.pt")
    pass
