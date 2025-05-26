# causal-synthetic-data-gen

**causal-synthetic-data-gen** is a Python-based tool for generating **synthetic longitudinal causal datasets** with autoregressive temporal dynamics and customizable **causal structures**. It supports **counterfactual scenario generation** for benchmarking causal inference algorithms.

## 📑 Table of Contents
- [Features](#-features)
- [Installation](#-installation)
- [Usage Example](#-usage-example)
- [Output Structure](#-output-structure)
- [License](#-license)
- [Citation](#-citation)
- [Contributions](#-contributions)
- [Contact](#-contact)

## 🚀 Features

- Synthetic generation of treatment, outcome, and covariate time series.
- Configurable causal structures: Direct, Chain, and Confounder.
- Support for **counterfactual trajectory generation** with:
  - Pointwise interventions
  - Continuous interventions
  - Gradual interventions
- Flexible output: factual and counterfactual datasets.
- Ready for **benchmarking causal inference models** on temporal data.

## 📦 Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/angeruzzi/causal-synthetic-data-gen.git
cd causal-synthetic-data-gen
pip install -r requirements.txt
```

## ⚙️ Configuration

The generator is configured through a YAML file. Example:

```yaml
n: 100
t: 30
structure_type: "confounder"
nonlinear: true
seed: 42
intervention_type: "gradual"
phi_T: 0.8
phi_Y: 0.9
t_interv: 15
delta_T: 0.5
betas:
  ty: 1.2
  tx: 0.7
  xy: 0.9
  xt: 0.6
normalize: true
noise_T_range: [-0.05, 0.05]
noise_Y_range: [-0.05, 0.05]
noise_X_range: [-0.05, 0.05]
split:
  train: 0.7
  val: 0.15
  test: 0.15
```

## 📝 Usage Example

To generate data from a config file:

```bash
python run_generator.py config/generator.yaml
```

This will:
- Generate the synthetic data based on the config file
- Split it into train/val/test sets
- Save `.npz` files into the `dataset/` folder
- Save the full resolved config (with randomly generated parameters) as a YAML log


## 📂 Output Structure

The generator produces structured NumPy arrays saved in `.npz` format:

- `treatments`: factual treatment sequences, shape `(n_individuals, t)`
- `outcomes`: factual outcomes
- `covariates`: optional, only in `chain` and `confounder` modes
- `*_cf`: counterfactual versions of the above, if `t_interv` is defined
- `log.yaml`: contains all parameters used, including those randomly generated

Example files saved in `dataset/`:

```
synthetic_confounder_nonlinear_30p_70n_train.npz
synthetic_confounder_nonlinear_30p_15n_val.npz
synthetic_confounder_nonlinear_30p_15n_test.npz
synthetic_confounder_nonlinear_30p_100n_log.yaml
```


## 📜 License
This project is licensed under the MIT License.

## 📖 Citation

If you use this tool in your research, please cite:

> Angeruzzi, A. S., & Albertini, M. K. (2025). Longitudinal Synthetic Data Generation from Causal Structures.

 
## 🤝 Contributions
Contributions, issues, and feature requests are welcome. Feel free to submit a pull request or open an issue.


## 📫 Contact

- **Alessandro S. Angeruzzi** - alessandro@angeruzzi.com.br / alessandro.angeruzzi@ufu.br
- **M. K. Albertini** - albertini@ufu.br


