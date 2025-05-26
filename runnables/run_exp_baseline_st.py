import os
import sys
import numpy as np
from datetime import datetime

# Importa os modelos criados
from src.baselinemodels.LinearRegressionDual import LinearRegressionModelDual

# Importa funções utilitárias
from src.support.load import load_config
from src.baselinemodels.utils import sumarize_exp_stats, sumarize_run_stats

# Mapeia os modelos disponíveis
MODEL_CLASSES = {
    "LinearRegressionModelDual": LinearRegressionModelDual,
}


def load_dataset(data_type: str, t: int, n: int) -> tuple:
    """
    Carrega os dados do dataset salvo em formato .npz.

    Parâmetros:
        - data_type (str): Tipo dos dados.
        - t (int): Número de períodos temporais.
        - n (int): Número de indivíduos.

    Retorna:
        - tuple: (T, Y, C) sendo cada um um np.ndarray.
    """
    file_path = f"dataset/{data_type}_{t}p_{n}n_test.npz"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset não encontrado: {file_path}")

    data = np.load(file_path)
    T = data["treatments"]
    Y = data["outcomes"]
    C = data["covariates"] if "covariates" in data else None
    Tcf = data["treatments_cf"] if "treatments_cf" in data else None
    Ycf = data["outcomes_cf"] if "outcomes_cf" in data else None
    Ccf = data["covariates_cf"] if "covariates_cf" in data else None            
    return T, Y, C, Tcf, Ycf, Ccf


def run_experiment_stats(exp_config: dict):
    """
    Executa uma lista de experimentos conforme definido no arquivo de configuração.

    Parâmetros:
        - exp_config (dict): Dicionário contendo a configuração dos experimentos.
    """
    experiments = exp_config.get("experiments", [])
    all_experiment_results = []

    # Criando um identificador único para a execução
    id_run = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for i, exp in enumerate(experiments):
        data_type = exp["data_type"]
        n = exp["n"]
        t = exp["t"]
        prev = exp["prev"]        
        horizon = exp["horizon"]
        models = exp["models"]

        print(f"\n🚀 Executando experimento {i+1}: {data_type} | n={n}, t={t}, prev = {prev}, horizon={horizon} | Modelos: {models}")

        # Carregando os dados do dataset
        T, Y, C, Tcf, Ycf, Ccf = load_dataset(data_type, t, n)

        id_exp = f"{id_run}_{data_type}_{t}t_{prev}p_{horizon}h_{n}n"

        # Criando diretório específico para esse experimento
        exp_dir = f"baselinestrun/{id_exp}"
        os.makedirs(exp_dir, exist_ok=True)

        model_results = []

        for model_info in models:
            
            model_name = model_info["name"]
            params = model_info.get("params", {})

            if model_name not in MODEL_CLASSES:
                print(f"⚠️ Modelo '{model_name}' não encontrado. Pulando...")
                continue

            model_class = MODEL_CLASSES[model_name]

            # Criando e rodando o modelo
            model = model_class(
                treatments=T,
                outcomes=Y,
                covariates=C,
                treatments_cf=Tcf,
                outcomes_cf=Ycf,
                covariates_cf=Ccf,                
                prev=prev,
                horizon=horizon,
                data_type=data_type,
                path_work=exp_dir,
                **params
            )

            results = model.call()
            print(f"✅ Modelo {model_name} finalizado.")

            # Armazena os resultados do modelo
            model_results.append({
                "model_name": model_name,                        
                "mse_steps" : results["MSE step"],
                "rmse_steps": results["RMSE step"],
                "r2_steps"  : results["R2 step"],
                "mae_steps" : results["MAE step"],                                
                "mape_steps": results["MAPE step"],
                "pehe_steps": results["PEHE step"],                
            })

        # Chamando sumarize_exp_stats e armazenando os resultados formatados
        exp_summary = sumarize_exp_stats(exp, exp_dir, model_results, id_exp)
        all_experiment_results.extend(exp_summary)  # Adiciona os dados já formatados

    # Chamando sumarize_run_stats ao final da execução de todos os experimentos
    sumarize_run_stats(id_run, all_experiment_results)

    print("\nTodos os experimentos foram concluídos!")

# Executar os experimentos a partir do arquivo YAML
if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/exp_baseline_stats.yaml"
    config = load_config(config_path)
    run_experiment_stats(config)
