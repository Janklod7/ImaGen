import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import image
from matplotlib import pyplot
from torchvision import transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
import mne



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

DATA_PATH = "./mini_dataset/" #mabe not like this
TRAIN_SIZE = 80
TEST_SIZE = 10
VALID_SIZE=10


def load_local_data():

    train = []
    test = []
    valid = []
    for i in range(TRAIN_SIZE):
        im = image.imread(DATA_PATH+str(i).zfill(5)+".PNG")
        train.append(im)
    for i in range(TRAIN_SIZE,TRAIN_SIZE+TEST_SIZE):
        im = image.imread(DATA_PATH + str(i).zfill(5) + ".PNG")
        test.append(im)
    for i in range(TRAIN_SIZE+TEST_SIZE, VALID_SIZE+TRAIN_SIZE+TEST_SIZE):
        im = image.imread(DATA_PATH + str(i).zfill(5) + ".PNG")
        valid.append(im)
    train_dataset = MyDataset(train)
    test_dataset = MyDataset(test)
    valid_dataset = MyDataset(valid)
    train_loader = torch.utils.data.DataLoader(train_dataset,shuffle=True, batch_size=4, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=4, num_workers=0)
    valid_loader =torch.utils.data.DataLoader(valid_dataset, shuffle=True, batch_size=4, num_workers=0)
    return train_loader, test_loader, valid_loader


data_dir = 'dataset'


train_loader, test_loader, valid_loader = load_local_data()


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

DATA_PATH = "./mini_dataset/"
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



class Encoder(nn.Module):

    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):

    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3,
                               stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2,
                               padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


### Define the loss function
loss_fn = torch.nn.MSELoss()

### Define an optimizer (both for the encoder and the decoder!)
lr= 0.001

### Set the random seed for reproducible results
torch.manual_seed(0)

### Initialize the two networks
d = 4

#model = Autoencoder(encoded_space_dim=encoded_space_dim)
encoder = Encoder(encoded_space_dim=d,fc2_input_dim=128)
decoder = Decoder(encoded_space_dim=d,fc2_input_dim=128)
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

# Move both the encoder and the decoder to the selected device
encoder.to(device)
decoder.to(device)
decoder.to(device)

## Training function
# def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer,noise_factor=0.3):
#     # Set train mode for both the encoder and the decoder
#     encoder.train()
#     decoder.train()
#     train_loss = []
#     # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
#     for image_batch, _ in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
#         # Move tensor to the proper device
#         image_noisy = add_noise(image_batch,noise_factor)
#         image_noisy = image_noisy.to(device)
#         # Encode data
#         encoded_data = encoder(image_noisy)
#         # Decode data
#         decoded_data = decoder(encoded_data)
#         # Evaluate loss
#         loss = loss_fn(decoded_data, image_noisy)
#         # Backward pass
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # Print batch loss
#         print('\t partial train loss (single batch): %f' % (loss.data))
#         train_loss.append(loss.detach().cpu().numpy())
#
#     return np.mean(train_loss)

def train_epoch_den(encoder, decoder, device, dataloader, loss_fn, optimizer,noise_factor=0.3):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    print(dataloader)
    for image_batch in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_noisy =image_batch
        image_noisy = image_noisy.to(device)
        # Encode data
        encoded_data = encoder(image_noisy)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_noisy)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

### Testing function
def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data

def plot_ae_outputs(encoder,decoder,n=5):
    plt.figure(figsize=(10,4.5))
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img = test_dataset[i][0].unsqueeze(0).to(device)
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
         rec_img  = decoder(encoder(img))
      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      if i == n//2:
         ax.set_title('Reconstructed images')

plt.show()

num_epochs = 30
diz_loss = {'train_loss': [], 'val_loss': []}
for epoch in range(num_epochs):
    train_loss = train_epoch_den(encoder, decoder, device,
                             train_loader, loss_fn, optim)
    val_loss = test_epoch(encoder, decoder, device, test_loader, loss_fn)
    print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs, train_loss, val_loss))
    diz_loss['train_loss'].append(train_loss)
    diz_loss['val_loss'].append(val_loss)
    plot_ae_outputs(encoder, decoder, n=5)


# test_epoch(encoder, decoder, device, test_loader, loss_fn).item()
#
# # Plot losses
# plt.figure(figsize=(10,8))
# plt.semilogy(diz_loss['train_loss'], label='Train')
# plt.semilogy(diz_loss['val_loss'], label='Valid')
# plt.xlabel('Epoch')
# plt.ylabel('Average Loss')
# #plt.grid()
# plt.legend()
# #plt.title('loss')
# plt.show()
#
#
# def plot_reconstructed(decoder, r0=(-5, 10), r1=(-10, 5), n=12):
#     plt.figure(figsize=(20, 8.5))
#     w = 28
#     img = np.zeros((n * w, n * w))
#     for i, y in enumerate(np.linspace(*r1, n)):
#         for j, x in enumerate(np.linspace(*r0, n)):
#             z = torch.Tensor([[x, y]]).to(device)
#             x_hat = decoder(z)
#             x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
#             img[(n - 1 - i) * w:(n - 1 - i + 1) * w, j * w:(j + 1) * w] = x_hat
#     plt.imshow(img, extent=[*r0, *r1], cmap='gist_gray')
#
#
# plot_reconstructed(decoder, r0=(-1, 1), r1=(-1, 1))
#
# encoded_samples = []
# for sample in tqdm(test_dataset):
#     img = sample[0].unsqueeze(0).to(device)
#     label = sample[1]
#     # Encode image
#     encoder.eval()
#     with torch.no_grad():
#         encoded_img  = encoder(img)
#     # Append to list
#     encoded_img = encoded_img.flatten().cpu().numpy()
#     encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_img)}
#     encoded_sample['label'] = label
#     encoded_samples.append(encoded_sample)
# encoded_samples = pd.DataFrame(encoded_samples)
# encoded_samples