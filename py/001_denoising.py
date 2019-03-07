#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 4 2019
@author: toshiki.ishikawa
"""

import os
import sys
import gc

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary



class DnDNN(nn.Module):

    def __init__(self, inputs=4096, n_layers=3):
        super(DnDNN, self).__init__()
        self.inputs = inputs
        self.n_layers = n_layers

    # =========================================================================
    # Encoder
    # =========================================================================
    def encoder(self, x):
        layers = []
        layers.append(nn.Linear(self.inputs, self.inputs//(2**(self.n_layers-1))))
        layers.append(nn.ReLU(inplace=True))

        for i in range(self.n_layers-1):
            layers.append(nn.Linear(self.inputs//(2**(2+i)), self.inputs//(2**(3+i))))
            layers.append(nn.ReLU(inplace=True))

        _encoder = nn.Sequential(*layers)
        return _encoder(x)

    # =========================================================================
    # Decoder
    # =========================================================================
    def decoder(self, x):
        layers = []
        for i in range(self.n_layers-1):
            layers.append(nn.Linear(self.inputs//(2**(1+self.n_layers-i)), self.inputs//(2**(self.n_layers-i))))
            layers.append(nn.ReLU(inplace=True))  
        
        layers.append(nn.Linear(self.inputs//(2**(self.n_layers-1)), self.inputs))
        
        _decoder = nn.Sequential(*layers)
        return _decoder(x)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x