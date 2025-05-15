# causal-synthetic-data-gen

**causal-synthetic-data-gen** is a Python-based tool for generating **synthetic longitudinal causal datasets** with autoregressive temporal dynamics and customizable **causal structures**. It supports **counterfactual scenario generation** for benchmarking causal inference algorithms.

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
