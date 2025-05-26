import os

def sumarize_exp_stats(exp_config: dict, exp_dir: str, model_results: list, id_exp: str) -> list:
    """
    Gera um sumário dos resultados de um experimento específico e salva em um arquivo de texto.

    Parâmetros:
        - exp_config (dict): Configuração do experimento.
        - exp_dir (str): Caminho onde os resultados foram armazenados.
        - model_results (list): Lista de dicionários com os resultados dos modelos.
        - id_exp (str): Identificador único do experimento.

    Retorna:
        - list: Lista de resultados formatados para sumarize_run_stats.
    """
    summary_txt_path = os.path.join(exp_dir, "summary_exp.txt")

    prev_periods = exp_config['prev']

    # Criando o arquivo de sumário do experimento
    with open(summary_txt_path, "w") as f:
        f.write(f"===== CONFIGURAÇÕES DO EXPERIMENTO =====\n")
        f.write(f"ID Exp\t: {id_exp}\n")
        f.write(f"Data Type\t: {exp_config['data_type']}\n")
        f.write(f"Individuals (n)\t: {exp_config['n']}\n")
        f.write(f"Prev Periods\t: {prev_periods}\n")
        f.write(f"Horizon\t: {exp_config['horizon']}\n\n")

        # Escrevendo cabeçalho da tabela
        f.write("===== RESULTADOS =====\n")
        header = (
            f"{'Model':<25}\t"
            f"MSE Steps\tRMSE Steps\tR² Steps\tMAE Steps\tMAPE Steps\tPEHE Steps\n"
        )
        f.write(header)

        # Criando a lista de saída já formatada para sumarize_run_stats
        summary = []
        for model in model_results:
            model_name = model["model_name"]

            # Formatação das métricas por step
            mse_steps = "[" + ", ".join(f"{mse:.4f}" for mse in model["mse_steps"]) + "]"
            rmse_steps = "[" + ", ".join(f"{rmse:.4f}" for rmse in model["rmse_steps"]) + "]"            
            r2_steps = "[" + ", ".join(f"{r2:.4f}" for r2 in model["r2_steps"]) + "]"
            mae_steps = "[" + ", ".join(f"{mae:.4f}" for mae in model["mae_steps"]) + "]"
            mape_steps = "[" + ", ".join(f"{mape:.4f}" for mape in model["mape_steps"]) + "]"
            pehe_steps = "[" + ", ".join(f"{mape:.4f}" for mape in model["pehe_steps"]) + "]"            

            # Escrevendo linha com resultados no arquivo
            line = (
                f"{model_name:<25}\t"
                f"{mse_steps}\t{rmse_steps}\t{r2_steps}\t{mae_steps}\t{mape_steps}\t{pehe_steps}\n"
            )
            f.write(line)

            # Adicionando os dados à lista formatada
            summary.append([
                id_exp,
                exp_config["data_type"],
                prev_periods,
                exp_config["horizon"],
                exp_config["n"],
                model_name,
                mse_steps,
                rmse_steps,
                r2_steps,
                mae_steps,
                mape_steps,
                pehe_steps                
            ])

    print(f"Sumário salvo: {summary_txt_path}")
    return summary


def sumarize_run_stats(id_run: str, all_experiment_results: list):
    """
    Gera um sumário geral de todos os experimentos executados.

    Parâmetros:
        - id_run (str): Identificador único da execução.
        - all_experiment_results (list): Lista de listas com os resultados formatados de cada experimento.
    """
    summary_txt_path = f"baselinestrun/{id_run}_summary.txt"

    # Criando o arquivo de resumo geral
    with open(summary_txt_path, "w") as f:
        f.write("===== RESUMO GERAL DOS EXPERIMENTOS =====\n")
        f.write(f"ID Run: {id_run}\n\n")

        # Cabeçalho
        header = (
            f"{'id_exp':<25}\t{'data_type':<15}\t{'prev_p.':<8}\t{'horizon':<7}\t{'n':<6}\t{'model':<25}\t"
            f"{'MSE Steps':<20}\t{'RMSE Steps':<20}\t{'R² Steps':<20}\t{'MAE Steps':<20}\t{'MAPE Steps':<20}\t{'PEHE Steps':<20}\n"
        )
        f.write(header)

        # Escrevendo os dados diretamente da lista
        for row in all_experiment_results:
            id_exp, data_type, prev_periods, horizon, n, model_name, mse_steps, rmse_steps, r2_steps, mae_steps, mape_steps, pehe_steps = row
            line = (
                f"{id_exp:<25}\t{data_type:<15}\t{prev_periods:<8}\t{horizon:<7}\t{n:<6}\t{model_name:<25}\t"
                f"{mse_steps:<20}\t{rmse_steps:<20}\t{r2_steps:<20}\t{mae_steps:<20}\t{mape_steps:<20}\t{pehe_steps:<20}\n"
            )

            f.write(line)

    print(f"\n📄 Sumário geral salvo: {summary_txt_path}")
