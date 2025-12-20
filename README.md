# Explainable machine learning prediction of compressive strength in ternary industrial waste-based geopolymer grouting materials

This repository provides the experimental dataset and the machine learning implementation
used in the manuscript:

**“Explainable machine learning prediction of compressive strength in ternary industrial waste-based geopolymer grouting material”**

submitted to *Powder Technology*.

The purpose of this repository is to support transparency, reproducibility, and reuse of the
experimental data and modeling framework presented in the paper.

---

## 📌 Overview

This study investigates the compressive strength of ternary industrial waste-based geopolymer
grouting materials composed of **slag, red mud, and fly ash**. An interpretable machine learning
framework integrating Particle Swarm Optimization (PSO) and SHAP analysis is developed to:

- Predict compressive strength within a defined experimental design space  
- Identify key governing parameters and their nonlinear effects  
- Provide quantitative guidance for mix proportion optimization  

The repository contains:
- The experimental dataset used for model development and validation  
- The implementation of the PSO-optimized Backpropagation Neural Network (PSO-BPNN)  
- Scripts for model training, evaluation, and interpretability analysis  

---

## 📁 Repository Structure

-
---

## 🧪 Experimental Data Description

- **Number of base mix designs:** 27  
- **Curing ages:** 3, 7, 28, and 56 days  
- **Total experimental samples:** 575  
- **Hold-out experimental validation samples:** 108  

The hold-out validation dataset was **pre-defined at the experimental design stage** and was
**strictly excluded** from all stages of model training, hyperparameter optimization, and
internal validation. It is used exclusively for final out-of-sample performance evaluation
within the same experimental design space.

All experiments were conducted following **GB/T 17671-2021 (ISO method)**.

---

## 🧠 Machine Learning Framework

- **Models implemented:**  
  - Backpropagation Neural Network (BPNN)  
  - Random Forest (RF)  
  - XGBoost  
  - Convolutional Neural Network (CNN, comparative baseline)  

- **Optimization:**  
  - Particle Swarm Optimization (PSO) for hyperparameter tuning  
  - Multiple independent runs with different random seeds to assess robustness  

- **Interpretability:**  
  - SHAP (Shapley Additive Explanations) for feature contribution analysis  

- **Uncertainty Analysis:**  
  - Sobol variance-based global sensitivity analysis  
  - ±2% relative uncertainty assumed for all input variables  

---

## 🔁 Reproducibility

All scripts use fixed random seeds where applicable to ensure reproducibility of the reported
results. Detailed model architectures, PSO parameter settings, and evaluation metrics are
consistent with those described in the manuscript.

After acceptance of the manuscript, this repository will serve as the permanent public archive
for the data and code associated with the study.

---

## ⚠️ Scope and Limitations

- The models are **data-driven** and trained within a clearly defined experimental design space.  
- The learned relationships represent **statistical correlations**, not explicit thermodynamic
  or constitutive laws.  
- Model performance and conclusions are **system- and dataset-specific** and should not be
  interpreted as universally applicable to all geopolymer or alkali-activated materials.

Future work will explore integration with physics-informed and variationally consistent learning
frameworks as governing relations become available.

---

## 📄 License and Usage

This repository is provided for **academic and research purposes**.  
If you use the data or code, please cite the associated manuscript.

---

## 📬 Contact

For questions related to the dataset or implementation, please contact:

**Xinbiao Wu**  
Email: 1144248789@qq.com
