from parameters import *
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from matplotlib import image
from matplotlib import pyplot
from torchvision import transforms as transforms

class Custom_dataset(torch.utils.data.Dataset):

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


    train_dataset = Custom_dataset(train)
    validation_dataset = Custom_dataset(validation)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=4, num_workers=0)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, shuffle=True, batch_size=4, num_workers=0)
    return train_loader, validation_loader


if __name__ == "__main__":

    print("loading data began")

    train_loader, test_loader = load_local_data()

    print("loading data done")

    # Instantiate the model
    model = MODEL

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
    model.to(device)

    print("device used:", device)



    print("number of epochs: ", EPOCHS)

    print("\n*** began training ***\n")

    for epoch in range(1, EPOCHS + 1):

        # monitor training loss
        train_loss = 0.0

        loss_lst = []
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
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    # Batch of test images
    pyplot.imshow(images[0].permute(1, 2, 0))
    pyplot.show()
    after_model = model(images)
    pyplot.imshow(after_model[0].detach().permute(1, 2, 0))
    pyplot.show()





