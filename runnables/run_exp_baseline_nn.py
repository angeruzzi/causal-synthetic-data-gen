import os
import sys
import shutil
import numpy as np

from omegaconf import OmegaConf
from datetime import datetime

from src.baselinemodels.RNN import RnnED
from src.baselinemodels.LSTM import LstmED
from src.baselinemodels.GRU import GruED
from src.baselinemodels.BaseNnED import manual_prediction

from src.support.load import load_config
from src.data.dataset import SyntheticDatasetCollection
from src.support.utils_logs_nn import plot_loss, sumarize_exp, sumarize_run

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping


MODEL_CLASSES = {
    "RnnED": RnnED,
    "LstmED": LstmED,
    "GruED": GruED,
}


def run_experiment(model_class, syntheticDs,  params, config, rep_folder):

    # Criando e rodando o modelo
    model = model_class(
        args=config,
        dataset_collection=syntheticDs,
        **params
    )

    model.save_dir = rep_folder

    early_stopping = EarlyStopping(monitor="val_loss", patience=30, mode="min")
    callbacks = [early_stopping]
    mlf_logger = None

    trainer = Trainer(
        accelerator="auto",
        devices=1,
        logger=mlf_logger,
        max_epochs=config.exp.max_epochs,
        callbacks=callbacks,
        # terminate_on_nan=True,
        gradient_clip_val=config.model.max_grad_norm
    )

    # Treinamento
    print(f"Treinando: {rep_folder} ... (epochs={config.exp.max_epochs})")
    trainer.fit(model)

    #Métrica de Treino
    train_loss_value = float(trainer.callback_metrics.get("train_loss", float("nan")))
    val_loss_value   = float(trainer.callback_metrics.get("val_loss", float("nan")))

    # Métricas de Teste por step
    test_mse_steps  = []
    test_rmse_steps = []    
    test_mae_steps  = []
    test_r2_steps   = []
    test_mape_steps = []
    test_pehe_steps = []  

    # Teste
    trainer.test(model)

    # Executar contrafactual manualmente, se disponível
    if config.dataset.has_counterfactual:
        print("Executando predições contrafactuais...")

        # Loop sobre os indivíduos do conjunto de teste
        pehe_steps = []
        for idx in range(len(syntheticDs.test_f)):
            # Extração dos dados factuais
            batch_data = syntheticDs.test_f[idx]
            prev_treatments = batch_data["prev_treatments"]
            prev_outcomes = batch_data["prev_outcomes"]
            current_treatments_factual = batch_data["current_treatments"]
            current_treatments_cf = batch_data["current_treatments_cf"]

            # Se houver covariáveis
            prev_covariates = batch_data.get("prev_covariates", None)
            current_covariates = batch_data.get("current_covariates", None)
            current_covariates_cf = batch_data.get("current_covariates_cf", None)            

            # Predições factuais e contrafactuais
            y_hat_factual = manual_prediction(model, prev_treatments, prev_outcomes, current_treatments_factual,
                                            prev_covariates, current_covariates)
            y_hat_cf = manual_prediction(model, prev_treatments, prev_outcomes, current_treatments_cf,
                                    prev_covariates, current_covariates_cf)

            y_hat_factual = y_hat_factual.squeeze(-1)
            y_hat_cf = y_hat_cf.squeeze(-1)

            # Valor real do factual e contrafactual
            y_true_factual = batch_data["outcomes"].squeeze()
            y_true_cf = batch_data["outcomes_cf"].squeeze()

            # Cálculo do PEHE para cada step
            pehe_individual_steps = ((y_true_factual - y_true_cf) - (y_hat_factual - y_hat_cf)) ** 2
            pehe_steps.append(pehe_individual_steps.cpu().numpy())

        # Agregar PEHE médio por step
        pehe_steps = np.array(pehe_steps)  # shape: [n_individuos, horizon]
        test_pehe_steps = pehe_steps.mean(axis=0).tolist()  # shape: [horizon]

    for i in range(config.exp.projection_horizon):
        test_mse_steps.append(float(trainer.callback_metrics.get(f"test_mse_step_{i+1}", None)))
        test_rmse_steps.append(float(trainer.callback_metrics.get(f"test_rmse_step_{i+1}", None)))        
        test_mae_steps.append(float(trainer.callback_metrics.get(f"test_mae_step_{i+1}", None)))
        test_r2_steps.append(float(trainer.callback_metrics.get(f"test_r2_step_{i+1}", None)))
        test_mape_steps.append(float(trainer.callback_metrics.get(f"test_mape_step_{i+1}", None)))
 
    # Logs/Plots
    plot_loss(model, loss_type="train", artifacts_path=rep_folder)
    plot_loss(model, loss_type="val", artifacts_path=rep_folder)

    return {
        "train_loss": float(train_loss_value) if train_loss_value is not None else None,
        "val_loss": float(val_loss_value) if val_loss_value is not None else None,

        "test_mse_step": test_mse_steps,
        "test_rmse_step": test_rmse_steps,        
        "test_mae_step": test_mae_steps,
        "test_r2_step": test_r2_steps,
        "test_mape_step": test_mape_steps,
        "test_pehe_step": test_pehe_steps,

        "config": config,
        "syntheticDs": syntheticDs
    }

def run(exp_config: dict):

    """
    Executa uma lista de experimentos conforme definido no arquivo de configuração.

    Parâmetros:
        - exp_config (dict): Dicionário contendo a configuração dos experimentos.
    """

    data_hora_ini = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_groups = exp_config.get("experiment_groups", [])

    # Carrega config
    config_path = "config/nn.yaml"
    config = OmegaConf.load(config_path)

    results_list = []

    for group in experiment_groups:

        data_type = group["data_type"]
        t = group["t"]
        prev = group["prev"]
        ntrain = group["ntrain"]
        nval = group["nval"]
        ntest = group["ntest"]
        has_covariate = group["has_covariate"]
        has_counterfactual = group["has_counterfactual"]
        experiments = group["experiments"]
        projection_horizon = group["projection_horizon"]

        group_path = f"baselinennrun/{data_hora_ini}_{data_type}_{t}t_{prev}p_{projection_horizon}h_{ntrain}n"

        # Cria diretório
        if os.path.exists(group_path):
            shutil.rmtree(group_path)
        os.makedirs(group_path, exist_ok=True)

        # Monta caminhos para datasets, incluindo o parâmetro data_type
        source_data = {
            'train': f'dataset/{data_type}_{t}p_{ntrain}n_train.npz',
            'val':   f'dataset/{data_type}_{t}p_{nval}n_val.npz',
            'test':  f'dataset/{data_type}_{t}p_{ntest}n_test.npz'
        }

        # Cria dataset
        syntheticDs = SyntheticDatasetCollection(
            source_data=source_data, 
            prev=prev,
            projection_horizon=projection_horizon,
            has_covariate=has_covariate,
            has_counterfactual=has_counterfactual)
        syntheticDs.process_data_multi()

        config.model.dim_outcomes = syntheticDs.train_f.data['prev_outcomes'].shape[-1]
        config.model.dim_treatments = syntheticDs.train_f.data['prev_treatments'].shape[-1]    
        config.model.dim_covariate = (
            syntheticDs.train_f.data['prev_covariates'].shape[-1] if syntheticDs.has_covariate else 0
        )

        for exp_num, exp in enumerate(experiments, start=1):

            model_name = exp["name"]
            params = exp.get("params", {})

            if model_name not in MODEL_CLASSES:
                print(f"Modelo '{model_name}' não encontrado. Pulando...")
                continue

            model_class = MODEL_CLASSES[model_name]

            batch_train_size = exp["batch_train_size"]
            batch_val_size = exp["batch_val_size"]
            epochs = exp["epochs"]
            num_layer = exp["num_layer"]
            repeat = exp["repeat"]
            dropout_rate = 0.1

            # Sobrescreve parâmetros
            config.model.name = model_name
            config.model.batch_size = batch_train_size
            config.model.num_layer = num_layer
            config.model.dropout_rate = dropout_rate
            config.dataset.data_type = data_type
            config.dataset.val_batch_size = batch_val_size
            config.dataset.has_covariate = has_covariate
            config.dataset.has_counterfactual = has_counterfactual
            config.exp.max_epochs = epochs
            config.exp.projection_horizon = projection_horizon
            
            # Define a pasta de artefatos específica do experimento
            exp_folder = os.path.join(group_path, f"exp_{str(exp_num)}_{model_name}")
            os.makedirs(exp_folder, exist_ok=True)

            rep_results = []
            for rep_id in range(1, repeat + 1):

                # Define a pasta de artefatos específica da repetição
                rep_folder = os.path.join(exp_folder, str(rep_id))
                os.makedirs(rep_folder, exist_ok=True)

                out = run_experiment(model_class, syntheticDs,  params, config, rep_folder)
                
                rep_results.append({
                    "rep_id": rep_id,
                    "train_loss": out["train_loss"],
                    "val_loss":   out["val_loss"],
                    "test_mse_step": out["test_mse_step"],
                    "test_rmse_step": out["test_rmse_step"],
                    "test_mae_step": out["test_mae_step"],
                    "test_r2_step": out["test_r2_step"],
                    "test_mape_step": out["test_mape_step"],
                    "test_pehe_step": out["test_pehe_step"],
                })
            
            mean_test_mse, mean_test_rmse,  mean_test_mae, mean_test_r2, mean_test_mape, mean_test_pehe = sumarize_exp(
                config=config,
                syntheticDs=syntheticDs,
                artifacts_path=exp_folder, # a pasta do experimento
                rep_results=rep_results    # todas as repetições
            )

            # Guarda as infos para o final_summary
            results_list.append({
                "exp_id": exp_num,
                "ref": data_hora_ini,
                "data_type": data_type,
                "t": t,
                "prev": prev,
                "projection_horizon": projection_horizon,
                "has_covariate": has_covariate,
                "has_counterfactual": has_counterfactual,
                "ntrain": ntrain,
                "name_model": model_name,
                "hidden_dim": config.model.hidden_dim,
                "num_layer": num_layer,
                "epochs": epochs,
                "dropout_rate": dropout_rate,
                "mean_test_mse": mean_test_mse,
                "mean_test_rmse": mean_test_rmse,                
                "mean_test_mae": mean_test_mae,
                "mean_test_r2": mean_test_r2,
                "mean_test_mape": mean_test_mape,
                "mean_test_pehe": mean_test_pehe,
            })

    data_hora_fim = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    sumarize_run(results_list, data_hora_ini, data_hora_fim, f"baselinennrun/{data_hora_ini}")

# Executar os experimentos a partir do arquivo YAML
if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/exp_baseline_nn.yaml"
    config = load_config(config_path)
    run(config)
