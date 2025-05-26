import numpy as np

def split(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Divide os dados em conjuntos de treino, validação e teste por indivíduo.

    Parâmetros:
        data (dict): Dicionário com arrays de shape (n, t), como "treatments", "outcomes", etc.
        train_ratio (float): Proporção do conjunto de treino.
        val_ratio (float): Proporção do conjunto de validação.
        test_ratio (float): Proporção do conjunto de teste.
        seed (int): Semente para reprodutibilidade.

    Retorna:
        dict: {'train': ..., 'val': ..., 'test': ...}, com os mesmos campos de `data`
    """

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "As proporções devem somar 1"

    np.random.seed(seed)

    # Detecta número de indivíduos a partir da primeira série válida
    series_keys = [k for k, v in data.items() if isinstance(v, np.ndarray) and v.ndim >= 2]
    if not series_keys:
        raise ValueError("Nenhuma série temporal encontrada nos dados.")
    
    n_individuos = data[series_keys[0]].shape[0]
    indices = np.arange(n_individuos)
    np.random.shuffle(indices)

    train_end = int(train_ratio * n_individuos)
    val_end = train_end + int(val_ratio * n_individuos)

    # Inicializa conjuntos
    datasets = {"train": {}, "val": {}, "test": {}}

    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.ndim >= 2 and value.shape[0] == n_individuos:
            datasets["train"][key] = value[indices[:train_end]]
            datasets["val"][key] = value[indices[train_end:val_end]]
            datasets["test"][key] = value[indices[val_end:]]
        else:
            # Copia os metadados diretamente em todos os splits
            datasets["train"][key] = value
            datasets["val"][key] = value
            datasets["test"][key] = value

    return datasets
