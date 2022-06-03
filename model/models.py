'''
@File    :   models.py
@Author  :   Raphael R. Eguchi
@Modified:   
@Contact :   possu@stanford.edu
@License :   (C)Copyright 2022, Raphael R. Eguchi, Stanford University.
@Desc    :   IgVAE Model.
'''

import torch.nn as nn
import torch
import numpy as np
from utils import *

class _Encoder(nn.Module):
    def __init__(self, nef, nc, nz):
        super(_Encoder, self).__init__()
        self.main = nn.Sequential(
            # Input: B x 1 x 512 x 512
            nn.Conv2d(1, nef * 2, 3, 1, 1, bias=True),
            nn.BatchNorm2d(nef * 2),
            nn.LeakyReLU(0.1, inplace=True),

            # 512 x 512
            nn.Conv2d(nef * 2, nef * 2, 3, 2, 1, bias=True),
            nn.BatchNorm2d(nef * 2),
            nn.LeakyReLU(0.1, inplace=True),

            # 256 x 256
            nn.Conv2d(nef * 2, nef * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(nef * 4),
            nn.LeakyReLU(0.1, inplace=True),

            # 128 x 128
            nn.Conv2d(nef * 4, nef * 8, 3, 2, 1, bias=True),
            nn.BatchNorm2d(nef * 8),
            nn.LeakyReLU(0.1, inplace=True),

            # 64 x 64
            nn.Conv2d(nef * 8, nef * 8, 3, 1, 1, bias=True),
            nn.BatchNorm2d(nef * 8),
            nn.LeakyReLU(0.1, inplace=True),

            # 64 x 64
            nn.Conv2d(nef * 8, nef * 8, 3, 2, 1, bias=True),
            nn.BatchNorm2d(nef * 8),
            nn.LeakyReLU(0.1, inplace=True),

            # 32 x 32
            nn.Conv2d(nef * 8, nef * 16, 3, 2, 1, bias=True),
            nn.BatchNorm2d(nef * 16),
            nn.LeakyReLU(0.1, inplace=True),

            # 16 x 16
            nn.Conv2d(nef * 16, nef * 16, 3, 1, 1, bias=True),
            nn.BatchNorm2d(nef * 16),
            nn.LeakyReLU(0.1, inplace=True),

            # 16 x 16
            nn.Conv2d(nef * 16, nef * 32, 4, 2, 1, bias=True),
            nn.BatchNorm2d(nef * 32),
            nn.LeakyReLU(0.1, inplace=True),

            # 8 x 8
            nn.Conv2d(nef * 32, nef * 32, 3, 2, 1, bias=True),
            nn.BatchNorm2d(nef * 32),
            nn.LeakyReLU(0.1, inplace=True),

            # 4 x 4
            nn.Conv2d(nef * 32, nc, 4, 1, 0, bias=True),
            nn.BatchNorm2d(nc),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Output: B x nc x 1 x 1
        )

        # Use FC to convert to nz x 1 x 1
        self.meanLayer = nn.Linear(nc, nz)
        self.varLayer = nn.Linear(nc, nz)

    def forward(self, input):
        output = self.main(input)
        output = output.view(output.size(0), -1) # Means and Variances derived from a common latent vector.
        means = self.meanLayer(output).unsqueeze(2).unsqueeze(3)
        logvars = self.varLayer(output).unsqueeze(2).unsqueeze(3)
        return means, logvars


class _Decoder(nn.Module):
    def __init__(self, ndf, nz):
        super(_Decoder, self).__init__()
        self.main = nn.Sequential(
           # Input is B x nz x 1 x 1
            nn.ConvTranspose2d(nz, ndf * 32, (4,1), (2,1), (1,0), bias=True),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.1, inplace=True),

            # 2 x 1
            nn.ConvTranspose2d(ndf * 32, ndf * 32, (4,1), (2,1), (1,0), bias=True),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.1, inplace=True),

            # 4 x 1
            nn.ConvTranspose2d(ndf * 32, ndf * 16, (4,1), (2,1), (1,0), bias=True),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.1, inplace=True),

            # 8 x 1
            nn.ConvTranspose2d(ndf * 16, ndf * 16, (3,1), (1,1), (1,0), bias=True),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.1, inplace=True),

            # 8 x 1
            nn.ConvTranspose2d(ndf * 16, ndf * 16, (3,1), (1,1), (1,0), bias=True),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.1, inplace=True),

            # 8 x 1
            nn.ConvTranspose2d(ndf * 16, ndf * 16, (4,1), (2,1), (1,0), bias=True),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.1, inplace=True),

            # 16 x 1
            nn.ConvTranspose2d(ndf * 16, ndf * 16, (3,1), (1,1), (1,0), bias=True),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.1, inplace=True),

            # 16 x 1
            nn.ConvTranspose2d(ndf * 16, ndf * 16, (3,1), (1,1), (1,0), bias=True),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.1, inplace=True),

            # 16 x 1
            nn.ConvTranspose2d(ndf * 16, ndf * 16, (4,1), (2,1), (1,0), bias=True),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.1, inplace=True),

            # 32 x 1
            nn.ConvTranspose2d(ndf * 16, ndf * 8, (4,1), (2,1), (1,0), bias=True),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.1, inplace=True),

            # 64 x 1
            nn.ConvTranspose2d(ndf * 8, ndf * 8, (4,1), (2,1), (1,0), bias=True),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.1, inplace=True),

            # 128 x 1
            nn.ConvTranspose2d(ndf * 8, ndf * 8, (3,1), (1,1), (1,0), bias=True),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.1, inplace=True),

            # 128 x 1
            nn.ConvTranspose2d(ndf * 8, ndf * 8, (3,1), (1,1), (1,0), bias=True),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.1, inplace=True),

            # 128 x 1
            nn.ConvTranspose2d(ndf * 8, ndf * 8, (4,1), (2,1), (1,0), bias=True),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.1, inplace=True),

            # 256 x 1
            nn.ConvTranspose2d(ndf * 8, ndf * 4, (4,1), (2,1), (1,0), bias=True),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.1, inplace=True),

            # 512 x 1
            nn.ConvTranspose2d(ndf * 4, 3, (3,1), (1,1), (1,0), bias=True),

            # Output: B x 3 x 512 x 1
        )

    def forward(self, input):
        coord = self.main(input).squeeze(3).permute(0,2,1)
        return coord

### Main VAE Class ###
class VAE(nn.Module):
    def __init__(self, nef=4, ndf=4, nc=512, nz=1024):
        super(VAE, self).__init__()
        self.encoder = _Encoder(nef, nc, nz)
        self.decoder = _Decoder(ndf, nz)

    def encode(self, x):
        means, logvars = self.encoder(x)
        return means, logvars

    def reparameterize(self, means, logvars, temp=1.0):
        std = torch.exp(0.5*logvars)
        eps = torch.randn_like(std)
        return means + eps*std * temp

    def forward(self, input,  mode='train', temp=1.0):
        if mode == 'train':
            dm = coords_to_dist(input)
            means, logvars = self.encode(dm)
            z = self.reparameterize(means, logvars)
            coord = self.decoder(z)
            return coord, means, logvars

        elif mode == 'recon':
            dm = coords_to_dist(input)
            means, logvars = self.encode(dm)
            z = self.reparameterize(means, logvars, temp=temp)
            coord = self.decoder(z)
            return coord

        elif mode == 'decode': 
            coord = self.decoder(input)
            return coord            

        elif mode == 'encode':
            dm = coords_to_dist(input)
            means, logvars = self.encoder(dm)
            z = self.reparameterize(means, logvars, temp=temp)
            return z

        else:
            raise Exception("Invalid Mode.")
