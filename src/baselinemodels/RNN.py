import torch.nn as nn
from src.baselinemodels.BaseNnED import BaseNnED

class RnnED(BaseNnED):

    def __init__(self, args, dataset_collection, **params):

        super().__init__(args, dataset_collection)

        self.encoder = nn.RNN(
            input_size=self.encoder_input_dim,  # entrada: prev_treatments + prev_outcomes (+ covariates)
            hidden_size=self.hidden_dim,
            num_layers=self.num_layer,
            batch_first=True
        )

        self.decoder = nn.RNN(
            input_size=self.decoder_input_dim,  # entrada: current_treatments (+ covariates)
            hidden_size=self.hidden_dim,
            num_layers=self.num_layer,
            batch_first=True
        )

        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
