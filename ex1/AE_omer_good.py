import parameters as p
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):

    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.name = "good old omer"

        # *************** Encoder **************** #

        self.conv1 = nn.Conv2d(3, 16, 3)

        self.conv2 = nn.Conv2d(16, 8, 4)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(8, 1, 3)

        self.convf = nn.Conv2d(2, 1, 1)

        self.lin1 = nn.Linear(30 * 30, 16 * 16)

        # *************** Decoder **************** #

        self.t_lin1 = nn.Linear(16 * 16, 30 * 30)

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

        return self.decoder_forward(self.encoder_forward(x), x.shape[0])

    def encoder_forward(self, x):

        # *************** Encoder **************** #

        p.printsize(1, x)

        x = p.printsize(2, F.leaky_relu(self.conv1(x), p.LEAKY_RATE))

        x = p.printsize(3, self.pool(x))

        x = p.printsize(4, F.leaky_relu(self.conv2(x), p.LEAKY_RATE))

        x = p.printsize(5, self.pool(x))

        x = p.printsize(6, F.leaky_relu(self.conv3(x), p.LEAKY_RATE))

        x = p.printsize(7, self.pool(x))

        x = p.printsize(8, torch.flatten(x, 1))

        x = p.printsize(9, F.leaky_relu(self.lin1(x), p.LEAKY_RATE))

        return x

    def decoder_forward(self, x, s):

        # *************** Decoder **************** #

        p.debug_print("---------------------",mode=p.DEBUG_SIZES)

        x = p.printsize(1, F.leaky_relu(self.t_lin1(x), p.LEAKY_RATE))

        x = p.printsize(2, torch.reshape(x, (s, 1, 30, 30)))

        x = p.printsize(3, F.leaky_relu(self.t_pool1(x), p.LEAKY_RATE))

        x = p.printsize(4, F.leaky_relu(self.t_conv1(x), p.LEAKY_RATE))

        x = p.printsize(5, F.leaky_relu(self.t_pool2(x), p.LEAKY_RATE))

        x = p.printsize(6, F.leaky_relu(self.t_conv2(x), p.LEAKY_RATE))

        x = p.printsize(7, F.leaky_relu(self.t_norm2(x), p.LEAKY_RATE))

        x = p.printsize(8, F.leaky_relu(self.t_pool3(x), p.LEAKY_RATE))

        x = p.printsize(9, torch.sigmoid(self.t_conv3(x)))

        p.printsize(10, x)

        return x
