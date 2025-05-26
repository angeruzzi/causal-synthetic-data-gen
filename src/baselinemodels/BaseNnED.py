import os
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from torchmetrics.functional.regression import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)


class BaseNnED(LightningModule):
    def __init__(self, args, dataset_collection):
        super().__init__()
        self.save_hyperparameters(args)
        self.dataset_collection = dataset_collection
        self.has_covariate = args.dataset.has_covariate
        self.hidden_dim = args.model.hidden_dim
        self.num_layer = args.model.num_layer
        self.projection_horizon = args.exp.projection_horizon

        self.encoder_input_dim: int = args.model.dim_treatments + args.model.dim_outcomes
        self.decoder_input_dim: int = args.model.dim_treatments

        if self.has_covariate:
            self.encoder_input_dim += args.model.dim_covariate
            self.decoder_input_dim += args.model.dim_covariate

        self.output_dim = args.model.dim_outcomes

        self.losses_dict = {"train": [], "val": []}

        self.model = None  # a ser definido pela subclasse
        self.fc = None     # a ser definido pela subclasse

    def build_encoder_input(self, batch):
        x = [batch["prev_treatments"], batch["prev_outcomes"]]
        if self.has_covariate:
            x.append(batch["prev_covariates"])
        return torch.cat(x, dim=-1)  # shape: [B, t, input_dim]

    def build_decoder_input(self, batch):
        x = [batch["current_treatments"]]
        if self.has_covariate:
            x.append(batch["current_covariates"])
        return torch.cat(x, dim=-1)  # shape: [B, h, decoder_input_dim]

    def forward(self, batch):
        # Encoder
        enc_x = self.build_encoder_input(batch)
        _, h_n = self.encoder(enc_x)

        # Decoder
        dec_x = self.build_decoder_input(batch)
        dec_out, _ = self.decoder(dec_x, h_n)

        y_hat = self.fc(dec_out)
        return y_hat

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        y_true = batch["outcomes"]
        loss = F.mse_loss(y_hat, y_true)
        return loss

    def training_epoch_end(self, outputs):
        epoch_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.losses_dict["train"].append(epoch_loss.item())
        self.log("train_loss", epoch_loss, on_epoch=True, prog_bar=True)  

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        y_true = batch["outcomes"]
        loss = F.mse_loss(y_hat, y_true)
        return loss

    def validation_epoch_end(self, losses):
        avg_loss = torch.stack(losses).mean()
        self.losses_dict["val"].append(avg_loss.item())
        self.log("val_loss", avg_loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        y_hat = self(batch)
        y_true = batch["outcomes"]
        return {"pred": y_hat, "target": y_true}

    def test_epoch_end(self, outputs):
        preds = torch.cat([x["pred"] for x in outputs], dim=0).squeeze(-1)
        targets = torch.cat([x["target"] for x in outputs], dim=0).squeeze(-1)
        h = preds.shape[1]

        for step in range(h):
            y_true = targets[:, step]
            y_pred = preds[:, step]

            self.log(f"test_mse_step_{step+1}" , mean_squared_error(preds[:, step], targets[:, step]))
            self.log(f"test_rmse_step_{step+1}", mean_squared_error(preds[:, step], targets[:, step], squared=False))            
            self.log(f"test_mae_step_{step+1}" , mean_absolute_error(preds[:, step], targets[:, step]))
            self.log(f"test_r2_step_{step+1}"  , r2_score(preds[:, step], targets[:, step]))
            self.log(f"test_mape_step_{step+1}", mean_absolute_percentage_error(preds[:, step], targets[:, step]))
    
            results_dict = {
                "preds": y_pred.tolist(),
                "targets": y_true.tolist()
            }

            # Plot (ordenação só para visualização)
            sorted_idx = y_true.argsort()
            y_true_sorted = y_true[sorted_idx]
            y_pred_sorted = y_pred[sorted_idx]

            plt.figure(figsize=(18, 8))
            x_indices = range(len(y_true_sorted))
            plt.scatter(x_indices, y_true_sorted.cpu().numpy(), label="Target", color="blue", alpha=0.7)
            plt.scatter(x_indices, y_pred_sorted.cpu().numpy(), label="Pred", color="orange", alpha=0.7)
            plt.title(f"Target vs. Pred (Teste) - Step {step+1}")
            plt.legend()
            plt.tight_layout()
            plt.grid()

            if hasattr(self, "save_dir") and self.save_dir is not None:
                plot_filename = os.path.join(self.save_dir, f"scatter_test_h{step+1}.png")
                plt.savefig(plot_filename)

                file_path = os.path.join(self.save_dir, f"preds_targets_h{step+1}.txt")
                with open(file_path, "w") as f:
                    f.write(str(results_dict))
            else:
                plt.show()
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.model.optimizer.learning_rate,
            weight_decay=self.hparams.model.optimizer.weight_decay,
        )

    def train_dataloader(self):
        return DataLoader(self.dataset_collection.train_f,
                          batch_size=self.hparams.model.batch_size,
                          shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_collection.val_f,
                          batch_size=self.hparams.model.batch_size,
                          shuffle=False, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.dataset_collection.test_f,
                          batch_size=self.hparams.model.batch_size,
                          shuffle=False, drop_last=False)


def manual_prediction(model, prev_treatments, prev_outcomes, current_treatments, 
                    prev_covariates=None, current_covariates=None):
    """
    Executa a previsão usando o modelo fornecido, dados de histórico (prev) e atuais (current).

    Retorna:
        - y_hat: predições [1, horizon, output_dim]
    """
    batch = {
        "prev_treatments": torch.tensor(prev_treatments).unsqueeze(0).float(),
        "prev_outcomes": torch.tensor(prev_outcomes).unsqueeze(0).float(),
        "current_treatments": torch.tensor(current_treatments).unsqueeze(0).float(),
    }

    if prev_covariates is not None and current_covariates is not None:
        batch["prev_covariates"] = torch.tensor(prev_covariates).unsqueeze(0).float()
        batch["current_covariates"] = torch.tensor(current_covariates).unsqueeze(0).float()

    # Modelo em modo de avaliação
    model.eval()
    with torch.no_grad():
        y_hat = model(batch)  # [1, horizon, output_dim]

    return y_hat.squeeze(0)  # [horizon, output_dim]
