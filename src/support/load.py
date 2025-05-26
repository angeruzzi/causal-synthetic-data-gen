import yaml

def load_config(config_path: str) -> dict:
    """
    Carrega um arquivo YAML contendo as configurações.

    Parâmetros:
        - config_path (str): Caminho do arquivo YAML.

    Retorna:
        - dict: Dicionário com a configuração.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
