# Ovarian Cancer Detection Project

## Codes and Data

This repository contains the codes and data for the following research article.

![SHAP](https://img.shields.io/badge/SHAP-v0.46.0-brightgreen) 
![Seaborn](https://img.shields.io/badge/Seaborn-v0.13.2-blue) 
![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.5.2-blue)
![NumPy](https://img.shields.io/badge/NumPy-v1.24.0-blue)
![Pandas](https://img.shields.io/badge/Pandas-v1.5.3-blue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-v3.6.2-blue)
![SciPy](https://img.shields.io/badge/SciPy-v1.10.0-blue)
![Flask](https://img.shields.io/badge/Flask-v2.2.2-blue)
![Langchain](https://img.shields.io/badge/Langchain-v0.0.100-orange)
![Hugging Face](https://img.shields.io/badge/HuggingFace-v4.22.1-red)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.13.0-orange)
![spaCy](https://img.shields.io/badge/spaCy-v3.4.1-green)
![NLTK](https://img.shields.io/badge/nltk-v3.7.0-blue)
![OpenAI](https://img.shields.io/badge/OpenAI-v0.27.0-blue)

## Project Overview

This project focuses on developing a machine learning-based diagnostic system for the reliable detection of ovarian cancer (OC), particularly in premenopausal and postmenopausal women. By leveraging explainable AI (XAI) tools like SHAP (SHapley Additive exPlanations), we aim to enhance the interpretability of model predictions, providing clinicians with valuable insights into individual risk factors.

## Workflow

![Workflow Diagram](images/image.png)

The workflow of the project consists of the following key steps:

1. **Data Preprocessing**: Cleaning and preparing the data for analysis.
2. **Feature Selection**: Utilizing Genetic Algorithms to identify important features that contribute to diagnosis.
3. **Model Training**: Employing the XGBoost algorithm for training the machine learning model.
4. **Evaluation**: Assessing the model's performance using ROC-AUC scores.

## Identified Biomarkers

The project identifies several key biomarkers associated with ovarian cancer, including:

- **HE4**
- **CA125**
- **CA19-9**
- **CA72-4**
- and several more biomarkers included.

These biomarkers have shown potential in improving diagnostic accuracy over existing methods.

## References

For detailed insights, refer to the following resources:

- **Research Paper**: [An ML-based decision support system for reliable diagnosis of ovarian cancer by leveraging explainable AI
](https://doi.org/10.1016/j.imu.2024.101553)


## Getting Started

### Prerequisites

- Python 3.12.4
- Required libraries: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `shap`, etc.

### Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/thanish11/ovarian_cancer_detection.git
cd ovarian_cancer_detection
pip install -r requirements.txt
