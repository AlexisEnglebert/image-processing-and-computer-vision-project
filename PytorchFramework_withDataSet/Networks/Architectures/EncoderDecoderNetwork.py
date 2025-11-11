    
import torch.nn as nn
import torch.nn.functional as F
import torch
#
# Architecture U-net pour la segmentation
#

"""
Loss at epoch 19: 0.46903614044703285
Validation Loss at i-th epoch:  0.5032406603467876
Loss curves graph saved to /home/alexis/Documents/dev/image-processing-and-computer-vision-project/PytorchFramework_withDataSet/Results/DefaultExp/loss_curves.pdf
The network is trained
Mean IoU: 0.3827549089005268
Class IoU: [0.66436311 0.12991206 0.43850193 0.29058624 0.39041122]
------------------------
Mean class F1: 0.5299695720979216
Class F1: [0.79833914 0.22995074 0.6096647  0.45031665 0.56157662]
------------------------
"""
class EncoderDecoderNet(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.nb_channel = param["MODEL"]["NB_CHANNEL"]

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, self.nb_channel, padding='same', kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_channel, self.nb_channel, padding='same', kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)
        self.encoder2 = nn.Sequential(
            nn.Conv2d(self.nb_channel, self.nb_channel*2, padding='same', kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_channel*2, self.nb_channel*2, padding='same', kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(self.nb_channel*2, self.nb_channel*4, padding='same', kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_channel*4, self.nb_channel*4, padding='same', kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.nb_channel*4, self.nb_channel*8, padding='same', kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_channel*8, self.nb_channel*8, padding='same', kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.up_sample1 = nn.ConvTranspose2d(self.nb_channel*8, self.nb_channel*4, kernel_size=2, stride=2)
        self.up_sample2 = nn.ConvTranspose2d(self.nb_channel*4, self.nb_channel*2, kernel_size=2, stride=2)
        self.up_sample3 = nn.ConvTranspose2d(self.nb_channel*2, self.nb_channel, kernel_size=2, stride=2)


        self.decode1 = nn.Sequential(
            nn.Conv2d(self.nb_channel * 8, self.nb_channel * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_channel * 4, self.nb_channel * 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decode2 = nn.Sequential(
            nn.Conv2d(self.nb_channel * 4, self.nb_channel * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_channel * 2, self.nb_channel * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decode3 = nn.Sequential(
            nn.Conv2d(self.nb_channel * 2, self.nb_channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_channel, self.nb_channel, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.final_block =  nn.Conv2d(self.nb_channel, 5, padding='same', kernel_size=1)
        
    def forward(self, x):

        skip_connections = [] # On sauvegardes les features pour chaque taille de channels
        #On descend dans le U (encoder)
        x = self.encoder1(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.encoder2(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.encoder3(x)
        skip_connections.append(x)
        x = self.pool(x)

        # On est dans le bottleneck (bas du U)
        x = self.bottleneck(x)
        # On monte dans le U (decoder)
        x = self.up_sample1(x)
        x = torch.cat([x, skip_connections[-1]], dim=1)
        x = self.decode1(x)

        x = self.up_sample2(x)
        x = torch.cat([x, skip_connections[-2]], dim=1)
        x = self.decode2(x)

        x = self.up_sample3(x)
        x = torch.cat([x, skip_connections[-3]], dim=1)
        x = self.decode3(x)

        # Maintenant on réduit à 5 channels
        x = self.final_block(x)

        return x
