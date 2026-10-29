# End-to-End Diabetes Prediction ML Project

An end-to-end Machine Learning pipeline to predict diabetes risk using diagnostic and clinical measurements.

---

## 1. Dataset Documentation (`data/diabetes.csv`)

### Educational & Synthetic Data Disclaimer
> [!IMPORTANT]
> **SYNTHETIC HEALTHCARE DATASET NOTICE**:
> This dataset has been formatted and generated specifically for **academic and educational purposes**.
> - It **does NOT contain real patient records**, protected health information (PHI), or personally identifiable information (PII).
> - It must **NOT be treated as clinical evidence** and cannot be used for medical diagnosis, clinical decision-making, or real-world treatment planning.

### Clinical Features Included

| Feature | Type | Range / Unit | Clinical Meaning |
| :--- | :--- | :--- | :--- |
| `Pregnancies` | Integer | 0 – 15 | Number of times pregnant (gestational stress indicator). |
| `Glucose` | Integer | 70 – 199 mg/dL | Plasma glucose concentration (2 hours in oral glucose tolerance test). |
| `BloodPressure` | Integer | 50 – 114 mm Hg | Diastolic blood pressure. |
| `SkinThickness` | Integer | 10 – 54 mm | Triceps skin fold thickness (body fat indicator). |
| `Insulin` | Integer | 20 – 420 $\mu$U/mL | 2-Hour serum insulin level. |
| `BMI` | Float | 18.5 – 52.0 $\text{kg/m}^2$ | Body Mass Index ($\text{weight in kg} / (\text{height in m})^2$). |
| `DiabetesPedigreeFunction` | Float | 0.08 – 2.30 | Genetic risk scoring based on family history of diabetes. |
| `Age` | Integer | 21 – 80 years | Patient age in completed years. |
| **`Outcome`** | Integer | **0 or 1** | **Target variable** (0 = Non-diabetic, 1 = Diabetic). |

---

## 2. Target Variable: Meaning & Limitations

- **`Outcome = 0` (Negative Class)**: Indicates low risk / non-diabetic status within the test window.
- **`Outcome = 1` (Positive Class)**: Indicates diabetes diagnosis or high clinical risk.

### Limitations of the Target:
1. **Binary Simplification**: In clinical practice, diabetes is a continuum (Normal $\rightarrow$ Prediabetes $\rightarrow$ Type 2 Diabetes). Binarizing into 0 and 1 simplifies classification but loses intermediate prediabetes nuances.
2. **Cross-Sectional Nature**: The label reflects a single diagnostic snapshot rather than longitudinal glycemic progression over time.
3. **Class Imbalance**: Like most healthcare scenarios, the positive class (Outcome = 1) is less frequent (~35%) than the negative class (~65%). Evaluation must account for this using metrics like Recall, F1-Score, and ROC-AUC rather than accuracy alone.

---

## 3. How to Inspect the Dataset

To inspect the dataset schema, data hygiene, and summary statistics, run:

```bash
python src/inspect_data.py
```
*(Ensure dependencies are installed: `pip install -r requirements.txt`)*


