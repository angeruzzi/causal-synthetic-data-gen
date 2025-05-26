# run_generate.py
import os
import sys
import yaml
import numpy as np
from datetime import datetime

from src.generator.csdg import generate_synthetic_data
from src.support.split_data import split
from src.support.load import load_config


def run_generation(config):

    # Separando os parâmetros obrigatórios da função principal
    name = config["name"]
    core_params = {
        "n": config["n"],
        "t": config["t"],
        "structure_type": config["structure_type"],
        "nonlinear": config["nonlinear"],
        "seed": config["seed"],
        "intervention_type": config["intervention_type"]
    }

    # Os demais parâmetros são passados via `config` para o argumento `config=...`
    generation_config = {
        k: v for k, v in config.items()
        if k not in core_params and k != "split"
    }

    # Gerar os dados sintéticos
    data = generate_synthetic_data(
        **core_params,
        config=generation_config
    )

    # Split dos dados
    split_ratios = config.get("split", {"train": 0.7, "val": 0.15, "test": 0.15})
    data_split = split(
        data,
        train_ratio=split_ratios["train"],
        val_ratio=split_ratios["val"],
        test_ratio=split_ratios["test"],
        seed=config["seed"]
    )

    # Diretório de destino
    os.makedirs("dataset", exist_ok=True)

    # Nome base dos arquivos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{name}_{core_params['structure_type']}_{'nonlinear' if core_params['nonlinear'] else 'linear'}_{core_params['t']}p_{core_params['n']}n"

    for split_key, split_data_dict in data_split.items():
        filename = f"{name}_{core_params['structure_type']}_{'nonlinear' if core_params['nonlinear'] else 'linear'}_{config["t"]}p_{int(config["n"]*split_ratios[split_key])}n_{split_key}.npz"
        filepath = os.path.join("dataset", filename)
        np.savez(filepath, **split_data_dict)

    # Atualizar config com valores efetivamente usados
    config.update({k: v for k, v in data.items() if k in [
        "structure_type", "intervention_type", "normalize", "nonlinear", "seed"
        "phi_T", "phi_Y", "delta_T", "t_interv", "betas", "effect_function",
        "effect_fx1", "effect_fx2", "noise_T_range", "noise_Y_range", "noise_X_range"
    ]})

    # Salvar log atualizado
    log_file = os.path.join("dataset", f"{base_name}_{timestamp}_log.yaml")
    with open(log_file, 'w') as log:
        yaml.dump(config, log, sort_keys=False)

    print(f"Dados gerados com base em {config_path}")
    print(f"Arquivos salvos com prefixo: {base_name}")

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/generator.yaml"
    config = load_config(config_path)
    run_generation(config)
