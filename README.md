# Heart Disease Prediction

A machine learning project that predicts **Heart Disease Status** (Yes/No) from a clinical + lifestyle dataset using a complete preprocessing pipeline and a **Logistic Regression** classifier.

## Repository contents
- **Notebook (code):** `ML_Mini_project.ipynb`
- **Project report (PDF):** `ML project report (9).pdf`

## Problem statement
Early identification of heart disease risk can support preventive care. This project builds a supervised ML model to classify whether a patient is likely to have heart disease based on demographic, vitals, and lifestyle indicators.

## Dataset
The notebook loads `heart_disease.csv` (uploaded in Google Colab) and performs cleaning and feature engineering.

**Target column:** `Heart Disease Status` (later standardized to `heart_disease_status`)

**Original columns (21):**
- Age
- Gender
- Blood Pressure
- Cholesterol Level
- Exercise Habits
- Smoking
- Family Heart Disease
- Diabetes
- BMI
- High Blood Pressure
- Low HDL Cholesterol
- High LDL Cholesterol
- Alcohol Consumption
- Stress Level
- Sleep Hours
- Sugar Consumption
- Triglyceride Level
- Fasting Blood Sugar
- CRP Level
- Homocysteine Level
- Heart Disease Status

## Workflow (Notebook)
1. **Data ingestion (Colab upload)**
2. **Cleaning & preprocessing**
   - Standardizes column names (lowercase, underscores)
   - Handles missing values
     - Numeric columns: converts to numeric, fills NaNs with mean, rounds to int
     - Categorical columns: trims whitespace, fills NaNs with mode
   - One-hot encodes categorical variables with `pd.get_dummies(..., drop_first=True)`
   - Exports cleaned dataset to `heart_disease_cleaned.csv`
3. **Train/test split + scaling**
   - Features: all columns except `heart_disease_status`
   - Standardization using `StandardScaler`
   - Split: 80% train / 20% test (`random_state=42`)
4. **Model training**
   - `LogisticRegression()`
5. **Evaluation & visualization**
   - Accuracy, confusion matrix, classification report
   - Confusion matrix heatmap
   - ROC curve + AUC
   - Histogram of predicted probabilities

## Results (from the notebook run)
The notebook prints an **accuracy around ~0.81** on the test set. The confusion matrix shown in the notebook indicates the model predicted only the majority class in that particular run, which causes precision/recall for the positive class to be 0.

> Note: This behavior often happens with **class imbalance** and/or default decision thresholds. Improvements may include class weighting (`class_weight='balanced'`), resampling (SMOTE), threshold tuning, and trying additional models.

## How to run
### Option 1: Google Colab (recommended)
1. Open `ML_Mini_project.ipynb` in Colab.
2. Run the notebook.
3. Upload `heart_disease.csv` when prompted by the upload widget.

### Option 2: Run locally (Jupyter)
1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
2. Place `heart_disease.csv` in the same directory as the notebook.
3. Run the notebook.

## Tech stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn

## Notes / Limitations
- The dataset file `heart_disease.csv` is not currently stored in this repository; it is uploaded at runtime in the notebook.
- The reported metrics can vary depending on the dataset distribution and preprocessing choices.

## License
Add a license if you plan to reuse or distribute this project.