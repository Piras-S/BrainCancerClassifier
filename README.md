# Brain Tumor Classification with Naive Bayes

This exploratory machine learning project investigates the classification of brain tumors using features extracted from MRI images. It focuses not only on building a predictive model, but also on evaluating how confident the model is in its decisions, a key aspect in sensitive domains like medicine.

---

## Dataset

The dataset contains:
- 700 samples
- 1500 numerical features (extracted from MRI scans)
- 4 tumor types:
  1. Pituitary Adenoma
  2. Germinoma
  3. Meningiomas
  4. Glioma

Source: [Kaggle – Brain Cancer Data](https://www.kaggle.com/datasets/michaelchalamet/brain-cancer-data)

---

## Approach

The project includes:

- Data exploration & visualization
- Feature selection using ANOVA F-statistic (`SelectKBest`)
- Naive Bayes classification (`GaussianNB`)
- Cross-validation to test generalizability
- Prediction probability analysis
- Probability calibration with `CalibratedClassifierCV`
- Visual inspection of model uncertainty

---

## Highlights

| Section | Description |
|--------|-------------|
| Feature Exploration | KDE plots to visualize distribution overlaps and redundancy |
| Feature Selection | Top 500 most informative features selected |
| Model | Gaussian Naive Bayes classifier |
| Cross-Validation | 5-fold CV for robustness |
| Calibration | Platt scaling (sigmoid) to correct overconfidence |
| Visuals | Confusion matrix, prediction bars, confidence histograms |

---

## Key Insight

> Without calibration, Naive Bayes often predicted probabilities close to 1.0 even on incorrect predictions.  
> Calibration helped make probabilities more realistic and trustworthy, especially important for risk-aware domains like healthcare.

---

## How to Run

1. Clone this repository  
2. Make sure you have the following dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
