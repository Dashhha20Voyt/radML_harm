# radML_harm
# Radiomics Harmonization Benchmark

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Comprehensive benchmark of harmonization methods for radiomics features across multi-center studies.

##  Features

- **3 Harmonization Methods**: ComBat, BART/RF, Riemannian
- **Comprehensive Evaluation Metrics**:
  - Batch leakage AUC (measures batch effect removal)
  - Bio-preservation AUC (biological signal retention)
  - ICC (Intraclass Correlation Coefficient)
- **Statistical Testing**: DeLong tests with bootstrap confidence intervals
- **Visualization**: UMAP plots before/after harmonization
- **Multiple Datasets**: Any dataset from radMLBench

##  Installation

### Requirements
- Python >= 3.8
- pip

### Installation Steps

1. Clone the repository:
git clone https://github.com/yourusername/radiomics-harmonization.git
cd radiomics-harmonization
2. Create virtual environment:
   python -m venv venv

# Activate on Linux/Mac:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
3.Install dependencies:
pip install -r requirements.txt
#  Quick Start
Run harmonization on any dataset:
python scripts/run_harmonization.py --dataset BraTS-2021 --output-dir results/brats2021
Available arguments:
--dataset: Dataset name from radMLBench (e.g., BraTS-2021, Arita2018)

--output-dir: Output directory for results (default: results/{dataset})

--n-batches: Number of pseudo-batches (default: 4)

--n-bootstrap: Bootstrap iterations (default: 1000)

--subset-size: UMAP subset size (default: 300)

Results structure:
results/{dataset}/
├── stats/                    # Statistical test results
│   ├── metrics.csv          # Main metrics
│   ├── delong_bio.csv       # DeLong test for bio-AUC
│   └── delong_batch.csv     # DeLong test for batch-AUC
└── plots/                    # Visualizations
    └── umap_visualization.png
## Methods
# ComBat
Empirical Bayes batch effect correction with robust error handling.

Reference: Johnson et al. (2007). "Adjusting batch effects in microarray expression data using empirical Bayes methods." Biostatistics, 8(1), 118-127.

# BART/RF
Random Forest-based harmonization modeling batch effects as covariates.

# Riemannian Harmonization (RiTex)
Geometric approach using Riemannian manifold alignment of covariance matrices.


📄 License
MIT License - see LICENSE file.

📧 Contact
Author: Darya Voitenko

Email: voitenko.da20@physics.msu.ru

GitHub: @Dashhha20Voyt

