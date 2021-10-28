from parameters import *
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

    def __init__(self, image_set, transform=transforms.ToTensor()):
        self.transform = transform
        self.image_set = image_set
        self.image_count = len(image_set)

    def __len__(self):
        return self.image_count

    def __getitem__(self, idx):
        img_tensor = self.transform(self.image_set[idx])
        return img_tensor

def load_local_data():
    train = []
    validation = []
    for i in range(VALIDATION_SIZE):
        im = image.imread(DATA_PATH + str(i).zfill(5) + ".PNG")
        validation.append(im)

    for i in range(VALIDATION_SIZE,VALIDATION_SIZE+TRAIN_SIZE):
        im = image.imread(DATA_PATH + str(i).zfill(5) + ".PNG")
        train.append(im)


    train_dataset = MyDataset(train)
    validation_dataset = MyDataset(validation)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=4, num_workers=0)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, shuffle=True, batch_size=4, num_workers=0)
    return train_loader, validation_loader


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # *************** Encoder **************** #

        self.conv1 = nn.Conv2d(3, 16, 3)

        self.conv2 = nn.Conv2d(16, 8, 4)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(8, 1, 3)

        self.convf = nn.Conv2d(2, 1, 1)

        self.lin1 = nn.Linear(30*30, 16*16)

        # *************** Decoder **************** #

        self.t_lin1 = nn.Linear(16*16, 30*30)

        self.t_pool0 = nn.ConvTranspose2d(1, 1, 1, stride=1)

        self.t_conv0 = nn.ConvTranspose2d(1, 1, 1, stride=2, output_padding=1)

        self.t_norm0 = nn.BatchNorm2d(2)

        self.t_pool1 = nn.ConvTranspose2d(1, 1, 2, stride=2)

        self.t_conv1 = nn.ConvTranspose2d(1, 8, 3, stride=1)

        self.t_pool2 = nn.ConvTranspose2d(8, 8, 2, stride=2)

        self.t_conv2 = nn.ConvTranspose2d(8, 16, 4, stride=1)

        self.t_pool3 = nn.ConvTranspose2d(16, 16, 2, stride=2)

        self.t_norm2 = nn.BatchNorm2d(16)

        self.t_conv3 = nn.ConvTranspose2d(16, 3, 3, stride=1)

    def forward(self, x):
        return self.decoder_forward(self.encoder_forward(x))

    def encoder_forward(self, x):
        # *************** Encoder **************** #

        printsize(1, x)

        x = printsize(2, F.leaky_relu(self.conv1(x), LEAKY_RATE))

        x = printsize(3, self.pool(x))

        x = printsize(4, F.leaky_relu(self.conv2(x), LEAKY_RATE))

        x = printsize(5, self.pool(x))

        x = printsize(6, F.leaky_relu(self.conv3(x), LEAKY_RATE))

        x = printsize(7, self.pool(x))

        x = printsize(8, torch.flatten(x, 1))

        x = printsize(9, F.leaky_relu(self.lin1(x), LEAKY_RATE))

        return x

    def decoder_forward(self, x):

        # *************** Decoder **************** #

        debug_print("---------------------")

        x = printsize(1, F.leaky_relu(self.t_lin1(x), LEAKY_RATE))

        x = printsize(2, torch.reshape(x,(4,1,30,30)))

        x = printsize(3, F.leaky_relu(self.t_pool1(x), LEAKY_RATE))

        x = printsize(4, F.leaky_relu(self.t_conv1(x), LEAKY_RATE))

        x = printsize(5, F.leaky_relu(self.t_pool2(x), LEAKY_RATE))

        x = printsize(6, F.leaky_relu(self.t_conv2(x), LEAKY_RATE))

        x = printsize(7, F.leaky_relu(self.t_norm2(x), LEAKY_RATE))

        x = printsize(8, F.leaky_relu(self.t_pool3(x), LEAKY_RATE))

        x = printsize(9, F.sigmoid(self.t_conv3(x)))

        printsize(10, x)

        return x


if __name__ == "__main__":

    print("loading data began")

    train_loader, test_loader = load_local_data()

    print("loading data done")

    # Instantiate the model
    model = ConvAutoencoder()

    # Loss function
    criterion = nn.MSELoss()  # L2 loss

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


    def get_device():
        if torch.cuda.is_available() and USE_GPU:
            device = 'cuda:0'
        else:
            device = 'cpu'
        return device


    device = get_device()
    print("device used:", device)
    model.to(device)

    # Epochs
    n_epochs = EPOCHS

    print("number of epochs: ", n_epochs)

    print("\n*** began training ***\n")

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
    pyplot.imshow(images[0].permute(1, 2, 0))
    pyplot.show()
    after_model = model(images)
    pyplot.imshow(after_model[0].detach().permute(1, 2, 0))
    pyplot.show()





