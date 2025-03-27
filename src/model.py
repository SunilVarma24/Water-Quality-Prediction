# src/model.py

import torch
import torch.nn as nn

class MultiTaskNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MultiTaskNN, self).__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Regression Head
        self.regression_head = nn.Linear(64, 1)

        # Classification Head
        self.classification_head = nn.Linear(64, num_classes)

    def forward(self, x):
        shared_out = self.shared_layers(x)
        reg_out = self.regression_head(shared_out)  # WQI prediction
        class_out = self.classification_head(shared_out)  # Classification
        return reg_out, class_out