import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data):
        self.images, self.labels = data

    def __getitem__(self, index):
        # Modify this to return a tuple of (image, label)
        return (self.images[index], self.labels[index])

    def __len__(self):
        return len(self.labels)

def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset

    trainimgs = torch.stack((torch.load(r'dtu_mlops\data\corruptmnist\train_images_0.pt'), 
                         torch.load(r'dtu_mlops\data\corruptmnist\train_images_1.pt'), 
                         torch.load(r'dtu_mlops\data\corruptmnist\train_images_2.pt'), 
                         torch.load(r'dtu_mlops\data\corruptmnist\train_images_3.pt'), 
                         torch.load(r'dtu_mlops\data\corruptmnist\train_images_4.pt'),
                         torch.load(r'dtu_mlops\data\corruptmnist\train_images_5.pt')), dim=1)
    # train = torch.randn(50000, 784)
    # test = torch.randn(10000, 784)
    trainimgs = trainimgs.view(-1, 1, 28, 28)
    trainimgs = trainimgs.squeeze()
    trainlabels = torch.stack((torch.load(r'dtu_mlops\data\corruptmnist\train_target_0.pt'),
                               torch.load(r'dtu_mlops\data\corruptmnist\train_target_1.pt'),
                               torch.load(r'dtu_mlops\data\corruptmnist\train_target_2.pt'),
                               torch.load(r'dtu_mlops\data\corruptmnist\train_target_3.pt'),
                               torch.load(r'dtu_mlops\data\corruptmnist\train_target_4.pt'),
                               torch.load(r'dtu_mlops\data\corruptmnist\train_target_5.pt')), dim=1)
    trainlabels = trainlabels.view(-1, 1)
    trainlabels = trainlabels.squeeze()
    train = MyDataset((trainimgs, trainlabels))
    train = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    # # concatenate trainimgs and trainlabels into a single tensor
    # train = torch.cat((trainimgs, trainlabels), dim=1)
    testimgs = torch.load(r'dtu_mlops\data\corruptmnist\test_images.pt')
    testlabels = torch.load(r'dtu_mlops\data\corruptmnist\test_target.pt')
    testimgs = testimgs.view(-1, 1, 28, 28)
    testimgs = testimgs.squeeze()
    testlabels = testlabels.view(-1, 1)
    testlabels = testlabels.squeeze()
    test = MyDataset((testimgs, testlabels))
    test = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)

    return train, test
