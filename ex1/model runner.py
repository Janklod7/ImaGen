from parameters import *
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
import numpy as np
import os
from datetime import datetime

from matplotlib import image
from matplotlib import pyplot
from torchvision import transforms as transforms


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, image_set, transform=transforms.ToTensor()):
        self.transform = transform
        self.image_set = image_set
        self.image_count = len(image_set)

    def __len__(self):
        return self.image_count

    def __getitem__(self, idxx):
        img_tensor = self.transform(self.image_set[idxx])
        return img_tensor

def present():
    idx = 5
    before = test_loader.dataset[idx].permute(1, 2, 0)
    a = model(test_loader.dataset[idx].to(device).unsqueeze(0))
    after = a[0].detach().cpu().permute(1, 2, 0)
    pyplot.imshow(np.concatenate((before,after), axis=1))
    pyplot.show()

def load_local_data():
    train = []
    validation = []
    for i in range(VALIDATION_SIZE):
        im = image.imread(DATA_PATH + str(i).zfill(5) + ".PNG")
        validation.append(im)

    for i in range(VALIDATION_SIZE, VALIDATION_SIZE+TRAIN_SIZE):
        im = image.imread(DATA_PATH + str(i).zfill(5) + ".PNG")
        train.append(im)

    train_dataset = CustomDataset(train)
    validation_dataset = CustomDataset(validation)
    train_loaderr = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=0)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=0)
    return train_loaderr, validation_loader


if __name__ == "__main__":

    print("loading data began")

    train_loader, test_loader = load_local_data()

    print("loading data done\n\n\n")

    # Instantiate the model
    model = MODEL

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


    def get_device():
        if torch.cuda.is_available() and USE_GPU:
            dev = 'cuda:0'
        else:
            dev = 'cpu'
        return dev


    device = get_device()
    model.to(device)

    print("device used:", device)

    print("number of epochs: ", EPOCHS)

    print("\n*** began training ***\n")
    loss_lst = []
    for epoch in range(1, EPOCHS + 1):

        if MIDDLE_STOP > 0 and epoch % MIDDLE_STOP == 0:
            if (input("present result [y/[n]]") == "y"):
                present()
            if (input("stop learning?? [y/[n]]") == "y"):
                break
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
        loss_lst.append(train_loss)
        add = ""
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    losses = pd.DataFrame(np.array(loss_lst))
    dir_name = f"{int((1-loss_lst[-1]) * 100)}%_{datetime.now().strftime('%H_%M_%S')}"
    os.mkdir(dir_name)
    losses.to_csv(f"{dir_name}/losses.csv")
    parameters = pd.DataFrame([("epoches", EPOCHS), ("batch size", BATCH_SIZE), ("learning rate", LEARNING_RATE),
                  ("model name", model.name)])
    parameters.to_csv(f"{dir_name}/parameters.csv")

    for i in range(1, 6):
        bef = test_loader.dataset[i].permute(1, 2, 0)
        a = model(test_loader.dataset[i].to(device).unsqueeze(0))
        af = a[0].detach().cpu().permute(1, 2, 0)
        pyplot.imshow(np.concatenate((bef, af), axis=1))
        pyplot.savefig(f"{dir_name}/images{i}.png")


    # Batch of test images
    present()
