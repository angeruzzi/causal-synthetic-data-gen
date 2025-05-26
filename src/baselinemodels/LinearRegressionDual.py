
import numpy as np
from sklearn.linear_model import LinearRegression

from src.baselinemodels.BaseStatisticalModel_counterfactual import BaseStatisticalModel

class LinearRegressionModelDual(BaseStatisticalModel):
    """
    Modelo de Regressão Linear baseado em BaseStatisticalModel.
    
    Implementa `call_individual` para realizar previsões utilizando regressão linear.
    """

    def call_individual(self, 
                        T_individuo: np.ndarray, 
                        C_individuo: np.ndarray, 
                        Y_individuo: np.ndarray, 
                        T_individuo_cf: np.ndarray = None, 
                        C_individuo_cf: np.ndarray = None, 
                        Y_individuo_cf: np.ndarray = None, 
                        prev: int = 1, 
                        horizon: int = 1) -> dict:
        """
        Executa a previsão para um indivíduo utilizando regressão linear,
        com suporte a contrafactual se fornecido.

        Retorna:
            - dict contendo train_X, train_y, test_X, test_y, predictions, 
              test_X_cf, test_y_cf, predictions_cf (se contrafactual fornecido).
        """

        assert horizon < len(T_individuo), "O horizonte de previsão não pode ser maior ou igual ao tamanho da sequência."

        # Factual - treinamento e teste
        if C_individuo is None:
            train_X = T_individuo[:-horizon].reshape(-1, 1)
            test_X = T_individuo[-horizon:].reshape(-1, 1)
        else:
            train_X = np.column_stack([T_individuo[:-horizon], C_individuo[:-horizon]])
            test_X = np.column_stack([T_individuo[-horizon:], C_individuo[-horizon:]])

        train_y = Y_individuo[:-horizon].reshape(-1, 1)
        test_y = Y_individuo[-horizon:]

        # Treina modelo factual
        model = LinearRegression()
        model.fit(train_X, train_y)

        # Predições factual
        predictions = [model.predict(test_X[i].reshape(1, -1))[0, 0] for i in range(horizon)]

        # Contrafactual se fornecido
        if T_individuo_cf is not None and Y_individuo_cf is not None:
            if C_individuo_cf is None:
                test_X_cf = T_individuo_cf[-horizon:].reshape(-1, 1)
            else:
                test_X_cf = np.column_stack([T_individuo_cf[-horizon:], C_individuo_cf[-horizon:]])

            test_y_cf = Y_individuo_cf[-horizon:]
            predictions_cf = [model.predict(test_X_cf[i].reshape(1, -1))[0, 0] for i in range(horizon)]
        else:
            # Se não houver contrafactual, define vazios
            test_X_cf = np.array([])
            test_y_cf = np.array([])
            predictions_cf = np.array([])

        return {
            "train_X": train_X,
            "train_y": train_y,
            "test_X": test_X,
            "test_y": test_y,
            "predictions": np.array(predictions),
            "test_X_cf": test_X_cf,
            "test_y_cf": test_y_cf,
            "predictions_cf": np.array(predictions_cf)
        }
