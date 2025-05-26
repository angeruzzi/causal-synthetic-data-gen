import os
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error


class BaseStatisticalModel(ABC):
    """
    Classe abstrata para modelos estatísticos para predição de resultados potenciais em dados longitudinais.

    Parâmetros:
        - treatments (np.ndarray): Matriz contendo os tratamentos de cada indivíduo.
        - outcomes (np.ndarray): Matriz contendo os resultados de cada indivíduo.
        - covariates (np.ndarray ou None): Matriz contendo covariáveis (opcional).
        - treatments_cf (np.ndarray): Matriz contendo os tratamentos contrafactuais de cada indivíduo.
        - outcomes_cf (np.ndarray): Matriz contendo os resultados contrafactuais de cada indivíduo.
        - covariates_cf (np.ndarray ou None): Matriz contendo covariáveis contrafactuais (opcional).        
        - horizon (int): Número de passos à frente para previsão.
        - data_type (str): Tipo dos dados.
        - path_work (str): Caminho do diretório de trabalho onde os arquivos serão salvos.
    """

    def __init__(self, 
                 treatments: np.ndarray, outcomes: np.ndarray, covariates: np.ndarray,
                 treatments_cf: np.ndarray, outcomes_cf: np.ndarray, covariates_cf: np.ndarray, 
                 prev: int, horizon: int, data_type: str, path_work: str, **params):
        self.treatments = treatments
        self.outcomes = outcomes
        self.covariates = covariates
        self.treatments_cf = treatments_cf
        self.outcomes_cf = outcomes_cf
        self.covariates_cf = covariates_cf        
        self.prev = prev
        self.horizon = horizon
        self.data_type = data_type
        self.path_work = path_work
        self.save_dir = None  # Será definido dentro do método `call`
        self.params = params

    @abstractmethod
    def call_individual(self, 
                        T_individuo: np.ndarray, C_individuo: np.ndarray, Y_individuo: np.ndarray, 
                        T_individuo_cf: np.ndarray, C_individuo_cf: np.ndarray, Y_individuo_cf: np.ndarray, 
                        prev: int, horizon: int) -> dict:
        """
        Método abstrato para previsão de um único indivíduo.

        Parâmetros:
            - T_individuo (np.ndarray): Tratamentos do indivíduo.
            - C_individuo (np.ndarray ou None): Covariáveis do indivíduo (pode ser None).
            - Y_individuo (np.ndarray): Resultados do indivíduo.
            - T_individuo_cf (np.ndarray): Tratamentos contrafactuais do indivíduo.
            - C_individuo_cf (np.ndarray ou None): Covariáveis contrafactuais do indivíduo (pode ser None).
            - Y_individuo_cf (np.ndarray): Resultados contrafactuais do indivíduo.            
            - prev (int): Número de períodos prévios
            - horizon (int): Número de passos à frente para previsão.

        Retorna:
            - dict: Dicionário contendo train_X, train_y, test_X, test_X_cf, test_y, test_y_cf, predictions, predictions_cf.
        """
        pass

    def call_predictions(self) -> dict:
        """
        Realiza previsões para todos os indivíduos e armazena os valores reais e preditos.

        Retorna:
            - dict: Contendo "pred" (lista de predições) e "target" (lista de valores reais).
        """
        all_preds, all_preds_cf, all_targets, all_targets_cf = [], [], [], []

        for individuo_idx in range(self.treatments.shape[0]):
            T_individuo = self.treatments[individuo_idx, :]
            Y_individuo = self.outcomes[individuo_idx, :]
            C_individuo = None if self.covariates is None else self.covariates[individuo_idx, :]
            
            T_individuo_cf = None if self.treatments_cf is None else self.treatments_cf[individuo_idx, :]
            Y_individuo_cf = None if self.outcomes_cf is None else self.outcomes_cf[individuo_idx, :]
            C_individuo_cf = None if self.covariates_cf is None else self.covariates_cf[individuo_idx, :]

            results = self.call_individual(T_individuo, C_individuo, Y_individuo, 
                                           T_individuo_cf, C_individuo_cf, Y_individuo_cf,
                                           self.prev, self.horizon)

            all_preds.append(results["predictions"])
            all_preds_cf.append(results["predictions_cf"])
            all_targets.append(results["test_y"])            
            all_targets_cf.append(results["test_y_cf"])

        return {
            "pred": np.array(all_preds), 
            "pred_cf": np.array(all_preds_cf), 
            "target": np.array(all_targets),
            "target_cf": np.array(all_targets_cf)
        }

    def call(self) -> dict:
        """
        Executa o pipeline completo:
        - Cria diretório para salvar resultados.
        - Realiza previsões.
        - Calcula MSE de cada step e final.
        - Gera gráficos e resumo do modelo.

        Retorna:
            - dict: Contendo MSE geral por step e MSE Final.
        """
        # Criando diretório para salvar resultados
        model_name = self.__class__.__name__
        self.save_dir = os.path.join(self.path_work, model_name)
        os.makedirs(self.save_dir, exist_ok=True)

        # Obtendo predições e alvos
        results = self.call_predictions()
        preds, preds_cf, targets, targets_cf = results["pred"], results["pred_cf"], results["target"], results["target_cf"]

        # Métricas por step
        mse_steps = []
        rmse_steps = []
        r2_steps = []
        mae_steps = []
        mape_steps = []
        pehe_steps = []

        has_counterfactuals = (self.outcomes_cf is not None)

        for step in range(self.horizon):
            y_true = targets[:, step]
            y_pred = preds[:, step]            
            mse_steps.append(mean_squared_error(y_true, y_pred))
            rmse_steps.append(root_mean_squared_error(y_true, y_pred))
            r2_steps.append(r2_score(y_true, y_pred))
            mae_steps.append(mean_absolute_error(y_true, y_pred))
            mape_steps.append(mean_absolute_percentage_error(y_true, y_pred))
            if has_counterfactuals:
                y_true_cf = targets_cf[:, step]
                y_pred_cf = preds_cf[:, step]          
                pehe_steps.append(self.calculate_pehe(y_true, y_true_cf, y_pred, y_pred_cf))

            self.plot_predictions(step, y_pred, y_true, self.save_dir)

        # Métricas gerais
        # mse_final = np.mean(mse_steps)
        # rmse_steps = np.mean(rmse_steps)
        # r2_final = np.mean(r2_steps)
        # mae_final = np.mean(mae_steps)
        # mape_final = np.mean(mape_steps)

        # Criando sumário do modelo
        summary_data = {
            "model_name": model_name,
            "n_test": self.treatments.shape[0],
            "data_type": self.data_type,
            "prev": self.prev,
            "horizon": self.horizon,
            "has_covariate": self.covariates is not None,
            "params": self.params,
            "metrics_steps": {
                "MSE": mse_steps,
                "RMSE": rmse_steps, 
                "R2": r2_steps,
                "MAE": mae_steps,
                "MAPE": mape_steps,
                "PEHE": pehe_steps,
            },
            # "metrics_final": {
            #     "MSE": mse_final,
            #     "R2": r2_final,
            #     "MAE": mae_final,
            #     "MAPE": mape_final
            # },
            "save_path": self.save_dir
        }

        self.summary_model(summary_data)
        self.save_preds_targets(preds, preds_cf, targets, targets_cf, self.save_dir)

        return {
            # "MSE geral": mse_final,
            # "R2 geral": r2_final,
            # "MAE geral": mae_final,
            # "MAPE geral": mape_final,
            "MSE step": mse_steps,
            "RMSE step": rmse_steps,
            "R2 step": r2_steps,
            "MAE step": mae_steps,
            "MAPE step": mape_steps,
            "PEHE step": pehe_steps
        }

    def plot_predictions(self, step: int, pred_list: np.ndarray, target_list: np.ndarray, save_dir: str):
        """
        Gera e salva um gráfico comparando predições e valores reais para um determinado step.

        Parâmetros:
            - step (int): Índice do step.
            - pred_list (np.ndarray): Lista de predições para o step.
            - target_list (np.ndarray): Lista de valores reais para o step.
            - save_dir (str): Diretório onde o gráfico será salvo.
        """
        sorted_idx = target_list.argsort()
        all_targs_sorted = target_list[sorted_idx]
        all_preds_sorted = pred_list[sorted_idx]

        # Criando o gráfico
        plt.figure(figsize=(18, 8))
        x_indices = range(len(all_targs_sorted))

        plt.scatter(x_indices, all_targs_sorted, label="Target", color="blue", alpha=0.7)
        plt.scatter(x_indices, all_preds_sorted, label="Pred", color="orange", alpha=0.7)

        plt.title(f"Target vs. Pred (Teste) - Step {step+1}")
        plt.legend()
        plt.tight_layout()
        plt.grid()

        filename = os.path.join(save_dir, f"scatter_test_h{step+1}.png")
        plt.savefig(filename)
        plt.close()

    def summary_model(self, summary_data: dict):
        """
        Gera um arquivo de resumo contendo informações do modelo.

        Parâmetros:
            - summary_data (dict): Dados do resumo do modelo.
        """
        summary_path = os.path.join(summary_data["save_path"], "summary_model.txt")
        with open(summary_path, "w") as f:
            f.write(f"Model\t\t: {summary_data['model_name']}\n")
            f.write(f"Data Type\t: {summary_data['data_type']}\n")
            f.write(f"Individuals\t: {summary_data['n_test']}\n")
            f.write(f"Prev Periods: {summary_data['prev']}\n")
            f.write(f"Horizon\t\t: {summary_data['horizon']}\n")
            f.write(f"Has Covar.\t: {summary_data['has_covariate']}\n")
            f.write(f"Parameters\t: {summary_data['params']}\n\n")            

            f.write("Step\tMSE\tRMSE\tR²\tMAE\tMAPE\tPEHE\n")
            # f.write(
            #     f"Geral\t{summary_data['metrics_final']['MSE']:.4f}".replace('.', ',')+"\t"
            #     f"{summary_data['metrics_final']['R2']:.4f}".replace('.', ',')+"\t"
            #     f"{summary_data['metrics_final']['MAE']:.4f}".replace('.', ',')+"\t"
            #     f"{summary_data['metrics_final']['MAPE']:.4f}".replace('.', ',')+"\n")
            
            for idx in range(summary_data['horizon']):
                f.write(
                    f"{idx+1}\t"
                    f"{summary_data['metrics_steps']['MSE'][idx]:.4f}".replace('.', ',')+"\t"
                    f"{summary_data['metrics_steps']['RMSE'][idx]:.4f}".replace('.', ',')+"\t"                    
                    f"{summary_data['metrics_steps']['R2'][idx]:.4f}".replace('.', ',')+"\t"
                    f"{summary_data['metrics_steps']['MAE'][idx]:.4f}".replace('.', ',')+"\t"
                    f"{summary_data['metrics_steps']['MAPE'][idx]:.4f}".replace('.', ',')+"\t"                    
                    f"{summary_data['metrics_steps']['PEHE'][idx]:.4f}".replace('.', ',')+"\n")

    def save_preds_targets(self, preds: np.ndarray, preds_cf: np.ndarray, targets: np.ndarray, targets_cf: np.ndarray, save_path: str):
        """
        Recebe arrays numpy de predições e valores reais, cria um dicionário com essas informações
        e salva em um arquivo .txt no diretório especificado em save_path.

        Parâmetros:
            - preds (np.ndarray): Array com valores preditos.
            - targets (np.ndarray): Array com valores reais.
            - save_path (str): Caminho do diretório onde o arquivo será salvo.
        """

        os.makedirs(save_path, exist_ok=True)  # Garante que o diretório exista

        # Cria o dicionário
        results_dict = {
            "preds": preds.tolist(),
            "preds_cf": preds_cf.tolist(),
            "targets": targets.tolist(),            
            "targets_cf": targets_cf.tolist()
        }

        # Salva o dicionário em formato txt
        file_path = os.path.join(save_path, "preds_targets.txt")
        with open(file_path, "w") as f:
            f.write(str(results_dict))

        print(f"✅ Arquivo salvo com sucesso em: {file_path}")

    def calculate_pehe(self, y_factual, y_cf, y_factual_pred, y_cf_pred):
        pehe_values = ((y_factual - y_cf) - (y_factual_pred - y_cf_pred)) ** 2
        return np.sqrt(np.mean(pehe_values))
