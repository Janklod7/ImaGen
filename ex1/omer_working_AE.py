import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import image
from matplotlib import pyplot
from torchvision import transforms as transforms

class MyDataset(torch.utils.data.Dataset):

    def __init__(self,image_set, transform=transforms.ToTensor()):
        self.transform = transform
        self.image_set = image_set
        self.image_count = len(image_set)

    def __len__(self):
        return self.image_count

    def __getitem__(self, idx):
        img_tensor = self.transform(self.image_set[idx])
        return img_tensor

DATA_PATH = "C:/Users/shach/Documents/mini_dataset/"
TRAIN_SIZE = 90
TEST_SIZE = 10

def load_local_data():

    train = []
    test = []
    for i in range(TRAIN_SIZE):
        im = image.imread(DATA_PATH+str(i).zfill(5)+".PNG")

        train.append(im)
    for i in range(TRAIN_SIZE,TRAIN_SIZE+TEST_SIZE):
        im = image.imread(DATA_PATH + str(i).zfill(5) + ".PNG")
        test.append(im)
    train_dataset = MyDataset(train)
    test_dataset = MyDataset(test)
    train_loader = torch.utils.data.DataLoader(train_dataset,shuffle=True, batch_size=4, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=4, num_workers=0)
    return train_loader, test_loader


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, 5, padding=1)
        self.conv2 = nn.Conv2d(16, 8, 5, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(8, 2, 4, padding=1)
        self.convf = nn.Conv2d(2, 1, 3, padding=1)
        #self.lin1 = nn.Linear(63*63*4, 4096)
        #self.lin2 = nn.Linear(4096, 256)
        # Decoder
        #self.t_lin1 = nn.Linear(256, 4096)
        #self.t_lin2 = nn.Linear(4096, 63*63*4)
        #self.t_project = nn.ConvTranspose2d(63*63*4, 4, 1, stride=2, bias=False)
        #self.t_project = nn.ConvTranspose2d(4,15876,1)
        self.t_conv0 = nn.ConvTranspose2d(1, 2, 3, stride=2)
        self.t_norm0 = nn.BatchNorm2d(2)
        self.t_conv1 = nn.ConvTranspose2d(2, 8, 4, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 5, stride=2)
        self.t_norm2 = nn.BatchNorm2d(16)
        self.t_conv3 = nn.ConvTranspose2d(16, 3, 5, stride=2)

    def forward(self, x):
        # encoder part
        print (x[0].size())
        x = F.relu(self.conv1(x))
        print(x[0].size())
        x = self.pool(x)
        print(x[0].size())
        x = F.relu(self.conv2(x))
        print(x[0].size())
        x = self.pool(x)
        print(x[0].size())
        x = F.relu(self.conv3(x))
        print(x[0].size())
        x = self.pool(x)
        print(x[0].size())
        x = F.relu(self.convf(x))
        print(x[0].size())
        x = self.pool(x)
        print(x[0].size())
        #x = torch.flatten(x, 1)
        #x = F.relu(self.lin1(x))
        #x = self.lin2(x)

        #decoder part
        #x = F.relu(self.t_lin1(x))
        #x = F.relu(self.t_lin2(x))
        #x = F.relu(self.t_project(x))
        print("---------------------")
        x = F.relu(self.t_conv0(x))
        print(x[0].size())
        x = F.relu(self.t_norm0(x))
        print(x[0].size())
        x = F.relu(self.t_conv1(x))
        print(x[0].size())
        x = F.relu(self.t_conv2(x))
        print(x[0].size())
        x = F.relu(self.t_norm2(x))
        print(x[0].size())
        x = F.sigmoid(self.t_conv3(x))
        print(x[0].size())

        return x



if __name__ == "__main__":
    print("begin")
    train_loader, test_loader = load_local_data()

    # Instantiate the model
    model = ConvAutoencoder()

    # Loss function
    criterion = nn.MSELoss() #L2 loss

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    def get_device():
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        return device


    device = get_device()
    print(device)
    model.to(device)

    # Epochs
    n_epochs = 2

    for epoch in range(1, n_epochs + 1):
        # monitor training loss
        train_loss = 0.0

        # Training
        for data in train_loader:
            images = data
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    # Batch of test images
    dataiter = iter(ts)
    images = dataiter.next()
    print(type(images[0]))
    pyplot.imshow(images[0].permute(1, 2, 0))
    pyplot.show()
    after_model = model(images)
    pyplot.imshow(after_model[0].detach().permute(1, 2, 0))
    pyplot.show()





