    
import torch.nn as nn
import torch.nn.functional as F
import torch
#
# Architecture U-net pour la reconstruction d'image self-supervised
#
class EncoderDecoderNet(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.nb_channel = param["MODEL"]["NB_CHANNEL"]
        self.dropout = nn.Dropout2d(p = 0.2)

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, self.nb_channel, padding="same", kernel_size=3),
            nn.BatchNorm2d(self.nb_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_channel, self.nb_channel, padding="same", kernel_size=3),
            nn.BatchNorm2d(self.nb_channel),
            nn.ReLU(inplace=True),
            self.dropout,
        )
        self.pool = nn.MaxPool2d(2,2)
        self.encoder2 = nn.Sequential(
            nn.Conv2d(self.nb_channel, self.nb_channel*2, padding="same", kernel_size=3),
            nn.BatchNorm2d(self.nb_channel*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_channel*2, self.nb_channel*2, padding="same", kernel_size=3),
            nn.BatchNorm2d(self.nb_channel*2),
            nn.ReLU(inplace=True),
            self.dropout,
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(self.nb_channel*2, self.nb_channel*4, padding="same", kernel_size=3),
            nn.BatchNorm2d(self.nb_channel*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_channel*4, self.nb_channel*4, padding="same", kernel_size=3),
            nn.BatchNorm2d(self.nb_channel*4),
            nn.ReLU(inplace=True),
            self.dropout,
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.nb_channel*4, self.nb_channel*8, padding="same", kernel_size=3),
            nn.BatchNorm2d(self.nb_channel*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_channel*8, self.nb_channel*8, padding="same", kernel_size=3),
            nn.BatchNorm2d(self.nb_channel*8),
            nn.ReLU(inplace=True),
        )


        #
        # https://discuss.pytorch.org/t/upsample-conv2d-vs-convtranspose2d/138081

        self.up_sample1 = nn.ConvTranspose2d(self.nb_channel*8, self.nb_channel*4, kernel_size=2, stride=2)
        self.up_sample2 = nn.ConvTranspose2d(self.nb_channel*4, self.nb_channel*2, kernel_size=2, stride=2)
        self.up_sample3 = nn.ConvTranspose2d(self.nb_channel*2, self.nb_channel, kernel_size=2, stride=2)


        self.decode1 = nn.Sequential(
            nn.Conv2d(self.nb_channel * 8, self.nb_channel * 4, 3, padding=1),
            nn.BatchNorm2d(self.nb_channel * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_channel * 4, self.nb_channel * 4, 3, padding=1),
            nn.BatchNorm2d(self.nb_channel * 4),
            nn.ReLU(inplace=True),
        )
        self.decode2 = nn.Sequential(
            nn.Conv2d(self.nb_channel * 4, self.nb_channel * 2, 3, padding=1),
            nn.BatchNorm2d(self.nb_channel * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_channel * 2, self.nb_channel * 2, 3, padding=1),
            nn.BatchNorm2d(self.nb_channel * 2),
            nn.ReLU(inplace=True),
        )
        self.decode3 = nn.Sequential(
            nn.Conv2d(self.nb_channel * 2, self.nb_channel, 3, padding=1),
            nn.BatchNorm2d(self.nb_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_channel, self.nb_channel, 3, padding=1),
            nn.BatchNorm2d(self.nb_channel),
            nn.ReLU(inplace=True),
        )
        # permet d'éviter l'overfitting :
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout2d.html
        # Note, for this model only 3 channels are in the final layer representing the value of each pixel
        self.final_block =  nn.Conv2d(self.nb_channel, 3, padding="same", kernel_size=1)
        
    def encode(self, x):
        skip_connections = []
        x = self.encoder1(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.encoder2(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.encoder3(x)
        skip_connections.append(x)
        x = self.pool(x)
        bottleneck_features = self.bottleneck(x)
        return bottleneck_features, skip_connections

    def decode(self, bottleneck_features, skip_connections):
        x = self.up_sample1(bottleneck_features)
        x = torch.cat([x, skip_connections[-1]], dim=1)
        x = self.decode1(x)

        x = self.up_sample2(x)
        x = torch.cat([x, skip_connections[-2]], dim=1)
        x = self.decode2(x)

        x = self.up_sample3(x)
        x = torch.cat([x, skip_connections[-3]], dim=1)
        x = self.decode3(x)

        x = self.final_block(x)
        return x

    def forward(self, x, return_features=False):
        bottleneck_features, skip_connections = self.encode(x)
        x = self.decode(bottleneck_features, skip_connections)
        if return_features:
            return x, bottleneck_features
        return x
