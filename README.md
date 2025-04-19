##### We have a problem with HR data which is the employees isn't satified so we go thorw the data trying to know where the unsatisfaction comes form.

###### Task 1: Definining Exploratory Data Analysis with an overview of the whole project and also problem framing.
###### Task 2: Importing libraries and Exploring the Dataset.
###### Task 3: Check for missing values.
###### Task 4: Creating visual methods to analyze the data.
###### Task 5: Prove that Monthly income is one of the problems.
###### Task 6: How to solve the problem.

# Task 2: Importing libraries and Exploring the Dataset.


```python
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
```


```python
#Read dataset
hrData=pd.read_csv("HR-Employee-Attrition.csv")
```


```python
#explore first 10 rows
hrData.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>DailyRate</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EmployeeCount</th>
      <th>EmployeeNumber</th>
      <th>...</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1102</td>
      <td>Sales</td>
      <td>1</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>279</td>
      <td>Research &amp; Development</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1373</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>2</td>
      <td>Other</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>2</td>
      <td>80</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>1392</td>
      <td>Research &amp; Development</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>5</td>
      <td>...</td>
      <td>3</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>591</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>1</td>
      <td>Medical</td>
      <td>1</td>
      <td>7</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>32</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>1005</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>8</td>
      <td>...</td>
      <td>3</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>3</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>59</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>1324</td>
      <td>Research &amp; Development</td>
      <td>3</td>
      <td>3</td>
      <td>Medical</td>
      <td>1</td>
      <td>10</td>
      <td>...</td>
      <td>1</td>
      <td>80</td>
      <td>3</td>
      <td>12</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>30</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>1358</td>
      <td>Research &amp; Development</td>
      <td>24</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>11</td>
      <td>...</td>
      <td>2</td>
      <td>80</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>38</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>216</td>
      <td>Research &amp; Development</td>
      <td>23</td>
      <td>3</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>12</td>
      <td>...</td>
      <td>2</td>
      <td>80</td>
      <td>0</td>
      <td>10</td>
      <td>2</td>
      <td>3</td>
      <td>9</td>
      <td>7</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>36</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>1299</td>
      <td>Research &amp; Development</td>
      <td>27</td>
      <td>3</td>
      <td>Medical</td>
      <td>1</td>
      <td>13</td>
      <td>...</td>
      <td>2</td>
      <td>80</td>
      <td>2</td>
      <td>17</td>
      <td>3</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 35 columns</p>
</div>




```python
#info about the shape of data and info about columns
hrData.info()

# Summary statistics
hrData.describe()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1470 entries, 0 to 1469
    Data columns (total 35 columns):
     #   Column                    Non-Null Count  Dtype 
    ---  ------                    --------------  ----- 
     0   Age                       1470 non-null   int64 
     1   Attrition                 1470 non-null   object
     2   BusinessTravel            1470 non-null   object
     3   DailyRate                 1470 non-null   int64 
     4   Department                1470 non-null   object
     5   DistanceFromHome          1470 non-null   int64 
     6   Education                 1470 non-null   int64 
     7   EducationField            1470 non-null   object
     8   EmployeeCount             1470 non-null   int64 
     9   EmployeeNumber            1470 non-null   int64 
     10  EnvironmentSatisfaction   1470 non-null   int64 
     11  Gender                    1470 non-null   object
     12  HourlyRate                1470 non-null   int64 
     13  JobInvolvement            1470 non-null   int64 
     14  JobLevel                  1470 non-null   int64 
     15  JobRole                   1470 non-null   object
     16  JobSatisfaction           1470 non-null   int64 
     17  MaritalStatus             1470 non-null   object
     18  MonthlyIncome             1470 non-null   int64 
     19  MonthlyRate               1470 non-null   int64 
     20  NumCompaniesWorked        1470 non-null   int64 
     21  Over18                    1470 non-null   object
     22  OverTime                  1470 non-null   object
     23  PercentSalaryHike         1470 non-null   int64 
     24  PerformanceRating         1470 non-null   int64 
     25  RelationshipSatisfaction  1470 non-null   int64 
     26  StandardHours             1470 non-null   int64 
     27  StockOptionLevel          1470 non-null   int64 
     28  TotalWorkingYears         1470 non-null   int64 
     29  TrainingTimesLastYear     1470 non-null   int64 
     30  WorkLifeBalance           1470 non-null   int64 
     31  YearsAtCompany            1470 non-null   int64 
     32  YearsInCurrentRole        1470 non-null   int64 
     33  YearsSinceLastPromotion   1470 non-null   int64 
     34  YearsWithCurrManager      1470 non-null   int64 
    dtypes: int64(26), object(9)
    memory usage: 402.1+ KB





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>DailyRate</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EmployeeCount</th>
      <th>EmployeeNumber</th>
      <th>EnvironmentSatisfaction</th>
      <th>HourlyRate</th>
      <th>JobInvolvement</th>
      <th>JobLevel</th>
      <th>...</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.0</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>...</td>
      <td>1470.000000</td>
      <td>1470.0</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>36.923810</td>
      <td>802.485714</td>
      <td>9.192517</td>
      <td>2.912925</td>
      <td>1.0</td>
      <td>1024.865306</td>
      <td>2.721769</td>
      <td>65.891156</td>
      <td>2.729932</td>
      <td>2.063946</td>
      <td>...</td>
      <td>2.712245</td>
      <td>80.0</td>
      <td>0.793878</td>
      <td>11.279592</td>
      <td>2.799320</td>
      <td>2.761224</td>
      <td>7.008163</td>
      <td>4.229252</td>
      <td>2.187755</td>
      <td>4.123129</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.135373</td>
      <td>403.509100</td>
      <td>8.106864</td>
      <td>1.024165</td>
      <td>0.0</td>
      <td>602.024335</td>
      <td>1.093082</td>
      <td>20.329428</td>
      <td>0.711561</td>
      <td>1.106940</td>
      <td>...</td>
      <td>1.081209</td>
      <td>0.0</td>
      <td>0.852077</td>
      <td>7.780782</td>
      <td>1.289271</td>
      <td>0.706476</td>
      <td>6.126525</td>
      <td>3.623137</td>
      <td>3.222430</td>
      <td>3.568136</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>102.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>30.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>80.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>30.000000</td>
      <td>465.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.0</td>
      <td>491.250000</td>
      <td>2.000000</td>
      <td>48.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>2.000000</td>
      <td>80.0</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>36.000000</td>
      <td>802.000000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>1.0</td>
      <td>1020.500000</td>
      <td>3.000000</td>
      <td>66.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>...</td>
      <td>3.000000</td>
      <td>80.0</td>
      <td>1.000000</td>
      <td>10.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>43.000000</td>
      <td>1157.000000</td>
      <td>14.000000</td>
      <td>4.000000</td>
      <td>1.0</td>
      <td>1555.750000</td>
      <td>4.000000</td>
      <td>83.750000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>...</td>
      <td>4.000000</td>
      <td>80.0</td>
      <td>1.000000</td>
      <td>15.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>9.000000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>60.000000</td>
      <td>1499.000000</td>
      <td>29.000000</td>
      <td>5.000000</td>
      <td>1.0</td>
      <td>2068.000000</td>
      <td>4.000000</td>
      <td>100.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>...</td>
      <td>4.000000</td>
      <td>80.0</td>
      <td>3.000000</td>
      <td>40.000000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>40.000000</td>
      <td>18.000000</td>
      <td>15.000000</td>
      <td>17.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 26 columns</p>
</div>



# Task 3: Check for missing values


```python
# Heatmap for missing values (if any)
plt.figure(figsize=(12, 6))
sns.heatmap(hrData.isnull(), cbar=False, cmap="Reds")
plt.title("Missing Values Heatmap")
plt.show()
```


    
![png](output_7_0.png)
    


There is no missing values

# Task 4: Creating visual methods to analyze the data.


```python
# Visualize realtions between numerical features
sns.pairplot(data=hrData)
plt.show()
```


    
![png](output_10_0.png)
    


After scanning this picture i noticed that there is something wrong with monthly income


```python
# Visualize value counts of categorical features
cat_cols = hrData.select_dtypes(include='object').columns
for col in cat_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=col, data=hrData, order=hrData[col].value_counts().index)
    plt.title(f'Count Plot of {col}')
    plt.tight_layout()
    plt.show()
```


    
![png](output_12_0.png)
    



    
![png](output_12_1.png)
    



    
![png](output_12_2.png)
    



    
![png](output_12_3.png)
    



    
![png](output_12_4.png)
    



    
![png](output_12_5.png)
    



    
![png](output_12_6.png)
    



    
![png](output_12_7.png)
    



    
![png](output_12_8.png)
    



```python
#trying to see where is the proble from count graphs
columns = [
    'JobSatisfaction', 'RelationshipSatisfaction', 'Attrition',
    'EnvironmentSatisfaction', 'MonthlyIncome',
    'MonthlyRate', 'OverTime', 'PercentSalaryHike',
    'WorkLifeBalance', 'YearsSinceLastPromotion','DistanceFromHome','Age'
]
fig, axes = plt.subplots(3, 4, figsize=(20, 12))
axes = axes.flatten()
for i, col in enumerate(columns):
    sns.histplot(data=hrData, x=col, ax=axes[i], kde=False)
    axes[i].set_title(col)

# Turn off the last empty subplot
for j in range(len(columns), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
```


    
![png](output_13_0.png)
    


From this draws i thik the monthly income is the reasone beacause the income is too low and too many people but not sure yet.

# Task 5: Prove that Monthly income is one of the problems


```python
#Trying to see the problem is with age or total working hours
sns.jointplot(x=hrData["MonthlyIncome"],y=hrData["TotalWorkingYears"],kind="scatter",hue=hrData['Department'])
sns.jointplot(x=hrData["MonthlyIncome"],y=hrData["Age"],kind="scatter",hue=hrData['Department'])
```




    <seaborn.axisgrid.JointGrid at 0x163d7f0f2f0>




    
![png](output_16_1.png)
    



    
![png](output_16_2.png)
    


here there is something wrong with monthly income there are workers that works for 20 years and take the same income as 1 to 5 years worker and this is not okay  "it is skewed" and the other graph ensures it by the graph of age so this is make more since now.


```python
# After knowing that the problem is with age we try to involve the job level with them to be more specific
sns.jointplot(x=hrData["MonthlyIncome"],y=hrData["Age"],kind="scatter",hue=hrData['JobLevel'])
```




    <seaborn.axisgrid.JointGrid at 0x163d68de5d0>




    
![png](output_18_1.png)
    


This is make more since now there is a big problem with monthly income and age and they are too old to be level 1 so we are sure now.


```python
# FacetGrid for Monthly Income vs Age by Job Level
g = sns.FacetGrid(hrData, col="JobLevel", height=4, aspect=1)
g.map_dataframe(sns.scatterplot, x="Age", y="MonthlyIncome", hue="Attrition", alpha=0.7)
g.add_legend()
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Monthly Income vs Age Across Job Levels')
plt.show()
```


    
![png](output_20_0.png)
    


there are a lot of people that has Attrition in level one and two


```python
# Trying to see the correlation throw heatmap to ensure my perspective
MonthlyIncomeData = hrData[['MonthlyIncome', 'TotalWorkingYears', 'Age', 'Department','JobLevel','YearsInCurrentRole','YearsAtCompany','YearsSinceLastPromotion','YearsWithCurrManager']].copy()
le = LabelEncoder()
MonthlyIncomeData['Department'] = le.fit_transform(MonthlyIncomeData['Department'])
MonthlyIncomeData['JobLevel'] = le.fit_transform(MonthlyIncomeData['JobLevel'])
plt.figure(figsize=(8, 6))
sns.heatmap(MonthlyIncomeData.corr(), annot=True, cmap='gray')
plt.title("Correlation Heatmap (with Encoded Department)")
plt.show()
```


    
![png](output_22_0.png)
    


Throw this heatmap now we are sure that you company need to reassign salaries and job level to be suitable with total working years and also i noticed from this correlation heatmap that there is also something wrong with Years at current role and level so this leads to low monthly income so this also a problem and also it lead to same thing that is age.

# Task 6: How to solve the problem.


```python
# here iam trying to know which roles that the monthly income is low
sns.barplot(x=hrData["MonthlyIncome"],y=hrData["JobRole"])
```




    <Axes: xlabel='MonthlyIncome', ylabel='JobRole'>




    
![png](output_25_1.png)
    


Throw this graph the income for hr , Reasearch scientist, laboratory technician and sales representative is tool low agains other employees.


```python
#this last heat map to see if there is any other problems that might cause the unsatisfaction problem
corr = hrData.corr(numeric_only=True)
plt.figure(figsize=(40, 30))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```


    
![png](output_27_0.png)
    


![image.png](e99c62d4-5051-45d1-bbb4-f3a8ff0b4c4d.png)

so throw this heat map we ensured that this is a problem with working years at the company, age, job level and monthly income that is whare the unsaticfaction comes from so you need to reasign level and income throw employess again throw a specific criteria which is workinng years to roles hr , Reasearch scientist, laboratory technician and sales representative.


```python

```
##### We have a problem with HR data which is the employees isn't satified so we go thorw the data trying to know where the unsatisfaction comes form.

###### Task 1: Definining Exploratory Data Analysis with an overview of the whole project and also problem framing.
###### Task 2: Importing libraries and Exploring the Dataset.
###### Task 3: Check for missing values.
###### Task 4: Creating visual methods to analyze the data.
###### Task 5: Prove that Monthly income is one of the problems.
###### Task 6: How to solve the problem.

# Task 2: Importing libraries and Exploring the Dataset.


```python
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
```


```python
#Read dataset
hrData=pd.read_csv("HR-Employee-Attrition.csv")
```


```python
#explore first 10 rows
hrData.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>DailyRate</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EmployeeCount</th>
      <th>EmployeeNumber</th>
      <th>...</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1102</td>
      <td>Sales</td>
      <td>1</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>279</td>
      <td>Research &amp; Development</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1373</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>2</td>
      <td>Other</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>2</td>
      <td>80</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>1392</td>
      <td>Research &amp; Development</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>5</td>
      <td>...</td>
      <td>3</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>591</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>1</td>
      <td>Medical</td>
      <td>1</td>
      <td>7</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>32</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>1005</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>8</td>
      <td>...</td>
      <td>3</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>3</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>59</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>1324</td>
      <td>Research &amp; Development</td>
      <td>3</td>
      <td>3</td>
      <td>Medical</td>
      <td>1</td>
      <td>10</td>
      <td>...</td>
      <td>1</td>
      <td>80</td>
      <td>3</td>
      <td>12</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>30</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>1358</td>
      <td>Research &amp; Development</td>
      <td>24</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>11</td>
      <td>...</td>
      <td>2</td>
      <td>80</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>38</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>216</td>
      <td>Research &amp; Development</td>
      <td>23</td>
      <td>3</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>12</td>
      <td>...</td>
      <td>2</td>
      <td>80</td>
      <td>0</td>
      <td>10</td>
      <td>2</td>
      <td>3</td>
      <td>9</td>
      <td>7</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>36</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>1299</td>
      <td>Research &amp; Development</td>
      <td>27</td>
      <td>3</td>
      <td>Medical</td>
      <td>1</td>
      <td>13</td>
      <td>...</td>
      <td>2</td>
      <td>80</td>
      <td>2</td>
      <td>17</td>
      <td>3</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 35 columns</p>
</div>




```python
#info about the shape of data and info about columns
hrData.info()

# Summary statistics
hrData.describe()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1470 entries, 0 to 1469
    Data columns (total 35 columns):
     #   Column                    Non-Null Count  Dtype 
    ---  ------                    --------------  ----- 
     0   Age                       1470 non-null   int64 
     1   Attrition                 1470 non-null   object
     2   BusinessTravel            1470 non-null   object
     3   DailyRate                 1470 non-null   int64 
     4   Department                1470 non-null   object
     5   DistanceFromHome          1470 non-null   int64 
     6   Education                 1470 non-null   int64 
     7   EducationField            1470 non-null   object
     8   EmployeeCount             1470 non-null   int64 
     9   EmployeeNumber            1470 non-null   int64 
     10  EnvironmentSatisfaction   1470 non-null   int64 
     11  Gender                    1470 non-null   object
     12  HourlyRate                1470 non-null   int64 
     13  JobInvolvement            1470 non-null   int64 
     14  JobLevel                  1470 non-null   int64 
     15  JobRole                   1470 non-null   object
     16  JobSatisfaction           1470 non-null   int64 
     17  MaritalStatus             1470 non-null   object
     18  MonthlyIncome             1470 non-null   int64 
     19  MonthlyRate               1470 non-null   int64 
     20  NumCompaniesWorked        1470 non-null   int64 
     21  Over18                    1470 non-null   object
     22  OverTime                  1470 non-null   object
     23  PercentSalaryHike         1470 non-null   int64 
     24  PerformanceRating         1470 non-null   int64 
     25  RelationshipSatisfaction  1470 non-null   int64 
     26  StandardHours             1470 non-null   int64 
     27  StockOptionLevel          1470 non-null   int64 
     28  TotalWorkingYears         1470 non-null   int64 
     29  TrainingTimesLastYear     1470 non-null   int64 
     30  WorkLifeBalance           1470 non-null   int64 
     31  YearsAtCompany            1470 non-null   int64 
     32  YearsInCurrentRole        1470 non-null   int64 
     33  YearsSinceLastPromotion   1470 non-null   int64 
     34  YearsWithCurrManager      1470 non-null   int64 
    dtypes: int64(26), object(9)
    memory usage: 402.1+ KB





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>DailyRate</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EmployeeCount</th>
      <th>EmployeeNumber</th>
      <th>EnvironmentSatisfaction</th>
      <th>HourlyRate</th>
      <th>JobInvolvement</th>
      <th>JobLevel</th>
      <th>...</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.0</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>...</td>
      <td>1470.000000</td>
      <td>1470.0</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>36.923810</td>
      <td>802.485714</td>
      <td>9.192517</td>
      <td>2.912925</td>
      <td>1.0</td>
      <td>1024.865306</td>
      <td>2.721769</td>
      <td>65.891156</td>
      <td>2.729932</td>
      <td>2.063946</td>
      <td>...</td>
      <td>2.712245</td>
      <td>80.0</td>
      <td>0.793878</td>
      <td>11.279592</td>
      <td>2.799320</td>
      <td>2.761224</td>
      <td>7.008163</td>
      <td>4.229252</td>
      <td>2.187755</td>
      <td>4.123129</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.135373</td>
      <td>403.509100</td>
      <td>8.106864</td>
      <td>1.024165</td>
      <td>0.0</td>
      <td>602.024335</td>
      <td>1.093082</td>
      <td>20.329428</td>
      <td>0.711561</td>
      <td>1.106940</td>
      <td>...</td>
      <td>1.081209</td>
      <td>0.0</td>
      <td>0.852077</td>
      <td>7.780782</td>
      <td>1.289271</td>
      <td>0.706476</td>
      <td>6.126525</td>
      <td>3.623137</td>
      <td>3.222430</td>
      <td>3.568136</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>102.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>30.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>80.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>30.000000</td>
      <td>465.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.0</td>
      <td>491.250000</td>
      <td>2.000000</td>
      <td>48.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>2.000000</td>
      <td>80.0</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>36.000000</td>
      <td>802.000000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>1.0</td>
      <td>1020.500000</td>
      <td>3.000000</td>
      <td>66.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>...</td>
      <td>3.000000</td>
      <td>80.0</td>
      <td>1.000000</td>
      <td>10.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>43.000000</td>
      <td>1157.000000</td>
      <td>14.000000</td>
      <td>4.000000</td>
      <td>1.0</td>
      <td>1555.750000</td>
      <td>4.000000</td>
      <td>83.750000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>...</td>
      <td>4.000000</td>
      <td>80.0</td>
      <td>1.000000</td>
      <td>15.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>9.000000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>60.000000</td>
      <td>1499.000000</td>
      <td>29.000000</td>
      <td>5.000000</td>
      <td>1.0</td>
      <td>2068.000000</td>
      <td>4.000000</td>
      <td>100.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>...</td>
      <td>4.000000</td>
      <td>80.0</td>
      <td>3.000000</td>
      <td>40.000000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>40.000000</td>
      <td>18.000000</td>
      <td>15.000000</td>
      <td>17.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 26 columns</p>
</div>



# Task 3: Check for missing values


```python
# Heatmap for missing values (if any)
plt.figure(figsize=(12, 6))
sns.heatmap(hrData.isnull(), cbar=False, cmap="Reds")
plt.title("Missing Values Heatmap")
plt.show()
```


    
![png](output_7_0.png)
    


There is no missing values

# Task 4: Creating visual methods to analyze the data.


```python
# Visualize realtions between numerical features
sns.pairplot(data=hrData)
plt.show()
```


    
![png](output_10_0.png)
    


After scanning this picture i noticed that there is something wrong with monthly income


```python
# Visualize value counts of categorical features
cat_cols = hrData.select_dtypes(include='object').columns
for col in cat_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=col, data=hrData, order=hrData[col].value_counts().index)
    plt.title(f'Count Plot of {col}')
    plt.tight_layout()
    plt.show()
```


    
![png](output_12_0.png)
    



    
![png](output_12_1.png)
    



    
![png](output_12_2.png)
    



    
![png](output_12_3.png)
    



    
![png](output_12_4.png)
    



    
![png](output_12_5.png)
    



    
![png](output_12_6.png)
    



    
![png](output_12_7.png)
    



    
![png](output_12_8.png)
    



```python
#trying to see where is the proble from count graphs
columns = [
    'JobSatisfaction', 'RelationshipSatisfaction', 'Attrition',
    'EnvironmentSatisfaction', 'MonthlyIncome',
    'MonthlyRate', 'OverTime', 'PercentSalaryHike',
    'WorkLifeBalance', 'YearsSinceLastPromotion','DistanceFromHome','Age'
]
fig, axes = plt.subplots(3, 4, figsize=(20, 12))
axes = axes.flatten()
for i, col in enumerate(columns):
    sns.histplot(data=hrData, x=col, ax=axes[i], kde=False)
    axes[i].set_title(col)

# Turn off the last empty subplot
for j in range(len(columns), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
```


    
![png](output_13_0.png)
    


From this draws i thik the monthly income is the reasone beacause the income is too low and too many people but not sure yet.

# Task 5: Prove that Monthly income is one of the problems


```python
#Trying to see the problem is with age or total working hours
sns.jointplot(x=hrData["MonthlyIncome"],y=hrData["TotalWorkingYears"],kind="scatter",hue=hrData['Department'])
sns.jointplot(x=hrData["MonthlyIncome"],y=hrData["Age"],kind="scatter",hue=hrData['Department'])
```




    <seaborn.axisgrid.JointGrid at 0x163d7f0f2f0>




    
![png](output_16_1.png)
    



    
![png](output_16_2.png)
    


here there is something wrong with monthly income there are workers that works for 20 years and take the same income as 1 to 5 years worker and this is not okay  "it is skewed" and the other graph ensures it by the graph of age so this is make more since now.


```python
# After knowing that the problem is with age we try to involve the job level with them to be more specific
sns.jointplot(x=hrData["MonthlyIncome"],y=hrData["Age"],kind="scatter",hue=hrData['JobLevel'])
```




    <seaborn.axisgrid.JointGrid at 0x163d68de5d0>




    
![png](output_18_1.png)
    


This is make more since now there is a big problem with monthly income and age and they are too old to be level 1 so we are sure now.


```python
# FacetGrid for Monthly Income vs Age by Job Level
g = sns.FacetGrid(hrData, col="JobLevel", height=4, aspect=1)
g.map_dataframe(sns.scatterplot, x="Age", y="MonthlyIncome", hue="Attrition", alpha=0.7)
g.add_legend()
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Monthly Income vs Age Across Job Levels')
plt.show()
```


    
![png](output_20_0.png)
    


there are a lot of people that has Attrition in level one and two


```python
# Trying to see the correlation throw heatmap to ensure my perspective
MonthlyIncomeData = hrData[['MonthlyIncome', 'TotalWorkingYears', 'Age', 'Department','JobLevel','YearsInCurrentRole','YearsAtCompany','YearsSinceLastPromotion','YearsWithCurrManager']].copy()
le = LabelEncoder()
MonthlyIncomeData['Department'] = le.fit_transform(MonthlyIncomeData['Department'])
MonthlyIncomeData['JobLevel'] = le.fit_transform(MonthlyIncomeData['JobLevel'])
plt.figure(figsize=(8, 6))
sns.heatmap(MonthlyIncomeData.corr(), annot=True, cmap='gray')
plt.title("Correlation Heatmap (with Encoded Department)")
plt.show()
```


    
![png](output_22_0.png)
    


Throw this heatmap now we are sure that you company need to reassign salaries and job level to be suitable with total working years and also i noticed from this correlation heatmap that there is also something wrong with Years at current role and level so this leads to low monthly income so this also a problem and also it lead to same thing that is age.

# Task 6: How to solve the problem.


```python
# here iam trying to know which roles that the monthly income is low
sns.barplot(x=hrData["MonthlyIncome"],y=hrData["JobRole"])
```




    <Axes: xlabel='MonthlyIncome', ylabel='JobRole'>




    
![png](output_25_1.png)
    


Throw this graph the income for hr , Reasearch scientist, laboratory technician and sales representative is tool low agains other employees.


```python
#this last heat map to see if there is any other problems that might cause the unsatisfaction problem
corr = hrData.corr(numeric_only=True)
plt.figure(figsize=(40, 30))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```


    
![png](output_27_0.png)
    


![image.png](e99c62d4-5051-45d1-bbb4-f3a8ff0b4c4d.png)

so throw this heat map we ensured that this is a problem with working years at the company, age, job level and monthly income that is whare the unsaticfaction comes from so you need to reasign level and income throw employess again throw a specific criteria which is workinng years to roles hr , Reasearch scientist, laboratory technician and sales representative.


```python

```
