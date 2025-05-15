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
git clone https://github.com/yourusername/causal-synthetic-data-gen.git
cd causal-synthetic-data-gen
pip install -r requirements.txt
```

## 📝 Usage Example
```python
from causal_synthetic_data_gen import SyntheticCausalGenerator
```

## 📂 Output Structure
The generator produces:

- treatments: factual treatment sequences
- outcomes: factual outcome sequences
- covariates: (optional) covariate sequences
- treatments_cf: counterfactual treatment sequences (if enabled)
- outcomes_cf: counterfactual outcome sequences (if enabled)


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


