import parameters as p
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):

    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # *************** Encoder **************** #

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, p.LAYER1, 3)

        self.conv2 = nn.Conv2d(p.LAYER1, p.LAYER2, 3)

        self.conv3 = nn.Conv2d(p.LAYER2, p.LAYER3, 3, padding=0)

        self.conv4 = nn.Conv2d(p.LAYER3, p.LAYER4, 2, padding=1)

        self.lin1 = nn.Linear(900, 256)
        self.lin2 = nn.Linear(30 * 30, 16 * 16)

        # *************** Decoder **************** #

        self.t_lin1 = nn.Linear(256, 900)
        self.t_lin2 = nn.Linear(30 * 30, 16 * 16)
        self.t_conv1 = nn.ConvTranspose2d(p.LAYER4, p.LAYER3, 2, stride=2, output_padding=1)
        self.t_norm1 = nn.BatchNorm2d(p.LAYER3)
        self.t_conv2 = nn.ConvTranspose2d(p.LAYER3, p.LAYER2, 3, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(p.LAYER2, p.LAYER1, 3, stride=2)
        self.t_norm2 = nn.BatchNorm2d(p.LAYER1)
        self.t_conv4 = nn.ConvTranspose2d(p.LAYER1, 3, 3, stride=2,output_padding=1)



    def forward(self, x):

        return self.decoder_forward(self.encoder_forward(x))

    def encoder_forward(self, x):

        # *************** Encoder **************** #

        p.printsize(1, x)

        x = p.printsize(2, F.leaky_relu(self.conv1(x), p.LEAKY_RATE))

        x = p.printsize(3, self.pool(x))

        x = p.printsize(4, F.leaky_relu(self.conv2(x), p.LEAKY_RATE))

        x = p.printsize(5, self.pool(x))

        x = p.printsize(6, F.leaky_relu(self.conv3(x), p.LEAKY_RATE))

        x = p.printsize(7, self.pool(x))

        x = p.printsize(8, F.leaky_relu(self.conv4(x), p.LEAKY_RATE))

        x = p.printsize(9, self.pool(x))

        #x = p.printsize(10, torch.flatten(x, 1))

        #x = p.printsize(11, F.leaky_relu(self.lin1(x), p.LEAKY_RATE))

        # x = p.printsize(11, F.leaky_relu(self.lin2(x), p.LEAKY_RATE))


        return x

    def decoder_forward(self, x):
        # *************** Decoder **************** #

        p.debug_print("\n---------------------\n", mode=p.DEBUG_SIZES)

        #x = p.printsize(1, F.leaky_relu(self.t_lin1(x), p.LEAKY_RATE))

        #x = p.printsize(2, F.leaky_relu(self.t_lin2(x), p.LEAKY_RATE))

        #x = p.printsize(2, torch.reshape(x, (p.BATCH_SIZE, 1, 30, 30)))

        x = p.printsize(3, F.leaky_relu(self.t_conv1(x), p.LEAKY_RATE))

        x = F.leaky_relu(self.t_norm1(x), p.LEAKY_RATE)

        x = p.printsize(4, F.leaky_relu(self.t_conv2(x), p.LEAKY_RATE))

        x = p.printsize(5, F.leaky_relu(self.t_conv3(x), p.LEAKY_RATE))

        x = F.leaky_relu(self.t_norm2(x), p.LEAKY_RATE)

        x = p.printsize(6, F.leaky_relu(self.t_conv4(x), p.LEAKY_RATE))

        p.printsize(7, x)

        return x
