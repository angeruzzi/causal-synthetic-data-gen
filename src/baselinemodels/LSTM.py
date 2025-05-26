import torch.nn as nn
from src.baselinemodels.BaseNnED import BaseNnED

class LstmED(BaseNnED):

    def __init__(self, args, dataset_collection, **params):

        super().__init__(args, dataset_collection)
        
        self.encoder = nn.LSTM(
            input_size=self.encoder_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layer,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=self.decoder_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layer,
            batch_first=True
        )

        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, batch):
        # Encoder
        enc_x = self.build_encoder_input(batch)
        _, (h_n, c_n) = self.encoder(enc_x)  # LSTM retorna (h_n, c_n)

        # Decoder
        dec_x = self.build_decoder_input(batch)
        dec_out, _ = self.decoder(dec_x, (h_n, c_n))

        y_hat = self.fc(dec_out)
        return y_hat  # shape: [batch, h, output_dim]
