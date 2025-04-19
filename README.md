
# HR Data Dissatisfaction Analysis

We have a problem with HR data which is that employees are not satisfied. So, we go through the data trying to know where the dissatisfaction comes from.

---

## Task 1: Exploratory Data Analysis & Problem Framing

We aim to analyze employee data to identify patterns related to dissatisfaction, particularly focusing on **monthly income** and its relationship with **job level**, **working years**, and **attrition**.

---

## Task 2: Importing Libraries & Exploring the Dataset

```python
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

hrData = pd.read_csv("HR-Employee-Attrition.csv")
hrData.head(10)
hrData.info()
hrData.describe()
```

---

## Task 3: Check for Missing Values

```python
plt.figure(figsize=(12, 6))
sns.heatmap(hrData.isnull(), cbar=False, cmap="Reds")
plt.title("Missing Values Heatmap")
plt.show()
```
> **No missing values found**

---

## Task 4: Visual Analysis

- Pairplot for numerical features
- Countplots for categorical features
- Histograms of key variables

```python
sns.pairplot(data=hrData)
```

> **Observation**: Monthly Income seems to have an unusual distribution, indicating possible issues.

---

## Task 5: Investigating Monthly Income Issues

- Scatter plots of Monthly Income vs TotalWorkingYears, Age
- Income vs Age with JobLevel
- Correlation heatmap

```python
sns.jointplot(x=hrData["MonthlyIncome"], y=hrData["TotalWorkingYears"], kind="scatter", hue=hrData['Department'])
```

> **Insight**: Employees with 20+ years experience earn similar to those with 1-5 years. Monthly income is **not aligned** with experience or age.

---

## Task 6: Proposed Solutions

```python
sns.barplot(x=hrData["MonthlyIncome"], y=hrData["JobRole"])
```

> **Roles affected**: HR, Research Scientist, Laboratory Technician, Sales Representative

```python
corr = hrData.corr(numeric_only=True)
plt.figure(figsize=(40, 30))
sns.heatmap(corr, annot=True, cmap='coolwarm')
```

---

## Conclusion

There’s a **clear mismatch** between job level, experience, and monthly income, leading to dissatisfaction. Re-evaluating **salary structure** based on **experience** and **role** is essential.
