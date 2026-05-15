# 🌾 Crop Recommendation System
### Using Soil and Climate Data

> **Predictive Analytics — Group Project**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-brightgreen)
![Accuracy](https://img.shields.io/badge/Accuracy-99.55%25-brightgreen)
![Models](https://img.shields.io/badge/Models-3-orange)

---

## 👥 Team Members

| Member | GitHub | Lead Stages |
|---|---|---|
| Adithyan | [@adithyanb276](https://github.com/adithyanb276) | S1, S3, S5, S6 |
| Abhita | [@abhitharajsr](https://github.com/abhitharajsr) | S2, S4, S7, S8, S9 |

---

## 🌐 Live App
### 👉 (https://crop-recommendation-system-acyz8kpnjckjjn4xbhapprk.streamlit.app/)

---

## 📌 Problem Statement

Farmers often lack access to data-driven guidance on which crop to grow based on their soil and climate conditions. Choosing the wrong crop leads to poor yield, financial loss, and inefficient use of resources like water and fertiliser.

This project builds an **AI-powered crop recommendation system** that predicts the most suitable crop based on 7 measurable soil and climate parameters — Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall.

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | [Kaggle — Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) |
| **Rows** | 2200 |
| **Columns** | 8 (7 features + 1 target) |
| **Classes** | 22 crops |
| **Balance** | Perfectly balanced — 100 samples per crop |
| **Missing Values** | None |
| **Duplicates** | None |

### Features
| Feature | Description | Unit |
|---|---|---|
| N | Nitrogen content in soil | kg/ha |
| P | Phosphorus content in soil | kg/ha |
| K | Potassium content in soil | kg/ha |
| temperature | Air temperature | °C |
| humidity | Relative humidity | % |
| ph | pH value of soil | 0–14 |
| rainfall | Annual rainfall | mm |

### 22 Crop Classes
apple · banana · blackgram · chickpea · coconut · coffee · cotton · grapes · jute · kidneybeans · lentil · maize · mango · mothbeans · mungbean · muskmelon · orange · papaya · pigeonpeas · pomegranate · rice · watermelon

---

## 🔬 Methodology — Data Science Life Cycle

| Stage | Description | Lead |
|---|---|---|
| S1 | Problem definition & literature review | Adithyan |
| S2 | Data collection & understanding | Abhitha |
| S3 | Data preprocessing & cleaning | Adithyan |
| S4 | Exploratory data analysis | Abhitha |
| S5 | Feature engineering & selection | Adithyan |
| S6 | Model building & training | Adithyan |
| S7 | Model evaluation & comparison | Abhitha |
| S8 | Model interpretation & explainability | Abhitha |
| S9 | Streamlit deployment | Abhitha |
| S10 | Documentation & presentation | Both |

---

## 🔧 Preprocessing

- ✅ Zero missing values — dataset is complete
- ✅ Zero duplicate rows — no data quality issues
- ⚠️ 611 outliers detected via IQR method — retained as they represent real agronomic conditions
- ✅ LabelEncoder applied to target column (22 crops → 0–21)
- ✅ StandardScaler applied — fit on training data only (prevents data leakage)
- ✅ Stratified 80/20 train-test split — 1760 train · 440 test

---

## 🧠 Models & Hyperparameter Tuning

All three models trained with **GridSearchCV (5-fold cross validation)**:

### 🥇 Random Forest
- Best params: `n_estimators=100, max_depth=10, min_samples_split=5`
- CV Accuracy: **99.60%** | Test Accuracy: **99.55%**

### 🥈 Decision Tree
- Best params: `criterion=gini, max_depth=15, min_samples_split=5`
- CV Accuracy: **98.58%** | Test Accuracy: **98.18%**

### 🥉 KNN
- Best params: `metric=manhattan, n_neighbors=5`
- CV Accuracy: **97.90%** | Test Accuracy: **97.73%**

---

## 📈 Results

| Model | CV Accuracy | Test Accuracy |
|---|---|---|
| 🥇 Random Forest | 99.60% | **99.55%** |
| 🥈 Decision Tree | 98.58% | 98.18% |
| 🥉 KNN | 97.90% | 97.73% |

### Key Findings
- 🌧️ **Humidity** is the most important feature (SHAP + RF importance agree)
- 🌿 **Rainfall** and **K (Potassium)** rank 2nd and 3rd
- 🔗 **P and K are highly correlated** (r = 0.74) — important multicollinearity finding
- ⚗️ **pH** has the least influence on crop prediction
- ✅ Dataset is perfectly balanced — 100 samples per crop, no SMOTE needed
- 🎯 Random Forest achieves near-perfect classification across all 22 crops

---

## 🔍 Model Explainability (SHAP)

SHAP (SHapley Additive exPlanations) values used to explain predictions:

- **SHAP Summary Plot**: Humidity dominates feature importance across all crop classes
- **SHAP Waterfall Plot**: Shows exactly why a specific crop was recommended
- Example: For Orange → K (+0.33), Humidity (+0.16), P (+0.15) were top drivers
- Example: For Rice → Rainfall (+0.24), N (+0.17), Humidity (+0.16) were top drivers

---

## 🖼️ App Screenshots

### Welcome Screen
![Welcome Screen](screenshots/welcome.png)

### Prediction — Rice with 90% confidence
![Prediction](screenshots/prediction.png)

### SHAP Explanation
![SHAP](screenshots/shap.png)

---

## ⚙️ Run Locally

```bash
# Clone the repository
git clone https://github.com/abhitharajsr/crop-recommendation-system.git
cd crop-recommendation-system

# Install dependencies
pip install -r requirements.txt

# Run the notebook first to generate model files
jupyter notebook crop_recommendation.ipynb

# Launch the Streamlit app
streamlit run app/app.py
```

---

## 📁 Repository Structure

```
crop-recommendation-system/
├── crop_recommendation.ipynb        ← Main notebook (all 10 stages)
├── app/
│   └── app.py                       ← Streamlit web app
├── data/
│   └── Crop_recommendation.csv      ← Dataset
├── models/                          ← Saved model files
│   ├── random_forest.pkl
│   ├── knn.pkl
│   ├── decision_tree.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── screenshots/                     ← App screenshots for README
├── individual_profiles/             ← GitHub activity screenshots
├── presentation/                    ← PPT presentation file
├── requirements.txt
└── README.md
```

---

## 🔗 Links

| Resource | Link |
|---|---|
| 🌐 Live App | [Streamlit Deployment](https://crop-recommendation-system-acyz8kpnjckjjn4xbhapprk.streamlit.app/) |
| 📓 Notebook | [crop_recommendation.ipynb](crop_recommendation.ipynb) |
| 📊 Dataset | [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) |
| 🐙 Repository | [GitHub](https://github.com/abhitharajsr/crop-recommendation-system) |

---

## 📚 References

1. Doshi, Z. et al. (2018). Agro Consultant: Intelligent Crop Recommendation System
2. Pudumalar, S. et al. (2017). Crop Recommendation System for Precision Agriculture
3. Mucherino, A. et al. (2009). K-Nearest Neighbour Classification of Agricultural Datasets
4. Lundberg, S. & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP)

---

*🌾 Predictive Analytics Group Project · Random Forest · KNN · Decision Tree · SHAP · Streamlit*
