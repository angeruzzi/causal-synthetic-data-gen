import os
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from typing import List, Dict, Optional

from src.baselinemodels.BaseNnED import BaseNnED
from src.data.dataset import SyntheticDatasetCollection

# #Gera os gráficos de LOSS
def plot_loss(model: BaseNnED, loss_type: str = "train", artifacts_path=None):
    if loss_type not in model.losses_dict:
        raise ValueError("...")
    losses = model.losses_dict[loss_type]

    plt.figure(figsize=(10, 6))
    plt.plot(losses, label=f'{loss_type.capitalize()} Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Evolução da {loss_type.capitalize()} Loss")
    plt.legend()
    plt.grid()

    if artifacts_path:
        plt.savefig(os.path.join(artifacts_path, f"{loss_type}_loss.png"))
    else:
        plt.show()
    
    plt.close()


def sumarize_exp(
    config: OmegaConf,
    syntheticDs:SyntheticDatasetCollection,
    artifacts_path: str,
    rep_results: Optional[List[Dict[str, float]]] = None
): 
    # --- Obtendo as informações do config ---
    name_model = config.model.name
    max_epochs = config.exp.max_epochs
    hidden_dim = config.model.hidden_dim
    num_layer = config.model.num_layer
    dropout_rate = config.model.dropout_rate
    batch_size = config.model.batch_size
    val_batch_size = config.dataset.val_batch_size
    data_type = config.dataset.data_type
    projection_horizon = config.exp.projection_horizon
    has_covariate = config.dataset.has_covariate
    has_counterfactual = config.dataset.has_counterfactual    
    
    # --- Quantidade de registros (amostras) ---
    n_train = len(syntheticDs.train_f)  # Tamanho do dataset de treino
    n_val   = len(syntheticDs.val_f)    # Tamanho do dataset de validação
    n_test  = len(syntheticDs.test_f)   # Tamanho do dataset de teste

    # --- Quantidade de períodos --- 
    time_periods = syntheticDs.train_f.data['active_entries'].shape[1]

    # --- Montando a string do resumo ---

    summary_text = f"""\
    ===== CONFIGURAÇÕES DO EXPERIMENTO =====
    name_model: {name_model}
    hidden_dim: {hidden_dim}
    num_layer : {num_layer}
    dropout_rate: {dropout_rate}
    max_epochs: {max_epochs}
    batch_size (train): {batch_size}
    val_batch_size: {val_batch_size}
    
    ===== TAMANHOS DOS DATASETS =====
    n_train: {n_train}
    n_val:   {n_val}
    n_test:  {n_test}
    data_type: {data_type}
    time_periods (T): {time_periods}
    projection_horizon: {projection_horizon}
    has_covariate: {has_covariate}
    has_counterfactual: {has_counterfactual}
    """
    
    summary_text += "\n===== MÉTRICAS FINAIS =====\n"

    if rep_results is not None and len(rep_results) > 0:
        # Cabeçalho da tabela
        summary_text += (
            "Rep\tTrain loss\tVal loss\t"
            "Test MSE Step\tTest RMSE Step\tTest MAE Step\tTest R2 Step\tTest MAPE Step\tTest PEHE Step\n"
        )

        valid_mse_step, valid_rmse_step, valid_mae_step, valid_r2_step, valid_mape_step, valid_pehe_step = [], [], [], [], [], []

        for item in rep_results:
            rep_id = item["rep_id"]
            train_loss = item["train_loss"]
            val_loss   = item["val_loss"]
            test_mse_step  = item.get("test_mse_step")
            test_rmse_step  = item.get("test_rmse_step")            
            test_mae_step  = item.get("test_mae_step")
            test_r2_step   = item.get("test_r2_step")                        
            test_mape_step = item.get("test_mape_step")
            test_pehe_step = item.get("test_pehe_step")            

            # Convertendo para string formatada
            train_loss_str = f"{train_loss:.4f}" if train_loss is not None else ""
            val_loss_str   = f"{val_loss:.4f}"   if val_loss   is not None else ""
            test_mse_step_str  = ", ".join(map(lambda x: f"{x:.4f}", test_mse_step)) if test_mse_step else ""
            test_rmse_step_str = ", ".join(map(lambda x: f"{x:.4f}", test_rmse_step)) if test_rmse_step else ""            
            test_mae_step_str  = ", ".join(map(lambda x: f"{x:.4f}", test_mae_step)) if test_mae_step else ""
            test_r2_step_str   = ", ".join(map(lambda x: f"{x:.4f}", test_r2_step)) if test_r2_step else ""
            test_mape_step_str = ", ".join(map(lambda x: f"{x:.4f}", test_mape_step)) if test_mape_step else ""
            test_pehe_step_str   = ", ".join(map(lambda x: f"{x:.4f}", test_pehe_step)) if test_pehe_step else ""

            summary_text += (
                f"{rep_id}\t{train_loss_str}\t{val_loss_str}\t"
                f"[{test_mse_step_str}]\t[{test_rmse_step_str}]\t[{test_mae_step_str}]\t[{test_r2_step_str}]\t[{test_mape_step_str}]\t[{test_pehe_step_str}]\n"
            )

            if test_mse_step: valid_mse_step.append(np.array(test_mse_step))
            if test_rmse_step: valid_rmse_step.append(np.array(test_rmse_step))            
            if test_mae_step: valid_mae_step.append(np.array(test_mae_step))
            if test_r2_step: valid_r2_step.append(np.array(test_r2_step))
            if test_mape_step: valid_mape_step.append(np.array(test_mape_step))
            if test_pehe_step: valid_pehe_step.append(np.array(test_pehe_step))

        # Médias por step
        def format_mean_step(metric_list):
            if len(metric_list) > 0:
                mean_metric = np.mean(np.stack(metric_list), axis=0)
                return ", ".join(f"{x:.4f}" for x in mean_metric)
            return ""

        format_mse 	= format_mean_step(valid_mse_step)
        format_rmse = format_mean_step(valid_rmse_step)
        format_mae 	= format_mean_step(valid_mae_step)
        format_r2 	= format_mean_step(valid_r2_step)
        format_mape = format_mean_step(valid_mape_step)
        format_pehe = format_mean_step(valid_pehe_step)

        summary_text += f"Mean Test MSE Step:  [{format_mse}]\n"
        summary_text += f"Mean Test RMSE Step: [{format_rmse}]\n"        
        summary_text += f"Mean Test MAE Step:  [{format_mae}]\n"
        summary_text += f"Mean Test R2 Step:   [{format_r2}]\n"
        summary_text += f"Mean Test MAPE Step: [{format_mape}]\n"
        summary_text += f"Mean Test PEHE Step: [{format_pehe}]\n"        

    else:
        summary_text += "Não há informações de repetições (rep_results) para exibir.\n"

    # --- Salvando no arquivo summary.txt dentro da pasta artifacts_path ---
    summary_file_path = os.path.join(artifacts_path, "summary.txt")
    with open(summary_file_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"As informações foram salvas em: {summary_file_path}")

    return format_mse, format_rmse, format_mae, format_r2, format_mape, format_pehe

def sumarize_run(results_list, data_hora_ini, data_hora_fim, run_path):

    final_summary_filename = f"{run_path}_final_sumary.tsv"
    with open(final_summary_filename, "w", encoding="utf-8") as f:
        
        f.write(f"ini: {data_hora_ini}\n")
        f.write(f"fim: {data_hora_fim}\n")

        f.write(
            "Exp\tRef\tDados\tprev_periods\thorizon\thas_covariate\thas_counterfactual\t"
            "Tam_Treino\tModel\tnum_hidden\tnum_layer\tEpochs\tdropout_rate\t"
            "Mean_Test_MSE\tMean_Test_RMSE\tMean_Test_MAE\tMean_Test_R2\tMean_Test_MAPE\tMean_Test_PEHE\n"
        )
        
        for item in results_list:
            exp_id = item["exp_id"]
            ref = item["ref"]
            data_type = item["data_type"]
            prev_p = item["prev"]
            projection_horizon = item["projection_horizon"]
            has_covariate = item["has_covariate"]
            has_counterfactual = item["has_counterfactual"]
            ntrain = item["ntrain"]
            name_model = item["name_model"]
            hidden_dim = item["hidden_dim"]
            num_layer = item["num_layer"]            
            epochs = item["epochs"]            
            dropout_rate = item["dropout_rate"]

            # Métricas
            mean_test_mse   = item.get("mean_test_mse")
            mean_test_rmse  = item.get("mean_test_rmse")         
            mean_test_mae   = item.get("mean_test_mae")
            mean_test_r2    = item.get("mean_test_r2")
            mean_test_mape  = item.get("mean_test_mape")
            mean_test_pehe  = item.get("mean_test_pehe")
            
            line = (
                f"{exp_id}\t{ref}\t{data_type}\t{prev_p}\t{projection_horizon}\t{has_covariate}\t{has_counterfactual}\t"
                f"{ntrain}\t{name_model}\t{hidden_dim}\t{num_layer}\t{epochs}\t{dropout_rate}\t"
                f"[{mean_test_mse}]\t[{mean_test_rmse}]\t[{mean_test_mae}]\t[{mean_test_r2}]\t[{mean_test_mape}\t[{mean_test_pehe}]\n"
            )

            f.write(line)
