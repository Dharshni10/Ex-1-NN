<H3>NAME : Dharshni V M</H3>
<H3>REGISTER NO : 212223240029 </H3>
<H3>EX. NO.1</H3>
<H3>DATE : </H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

```python
import pandas as pd                  
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
df = pd.read_csv("Churn_Modelling.csv")
print(df)
x = df.iloc[:, :-1].values
x
y = df.iloc[:, -1].values
y
print(df.isnull().sum())
df.duplicated()
df.describe()
df = df.drop(['Surname', 'Geography', 'Gender'], axis=1)
scaler = MinMaxScaler()
df1 = pd.DataFrame(scaler.fit_transform(df))
print(df1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))

```

## OUTPUT:
#### DATASET PREVIEW:

<img width="609" height="704" alt="image" src="https://github.com/user-attachments/assets/f3592e74-757f-45c4-a2cc-d5ef03260dda" />

#### FEATURE MATRIX:

<img width="557" height="132" alt="image" src="https://github.com/user-attachments/assets/009352e7-d7ff-4902-a55b-888d6091c30c" />

#### TARGET VECTOR:

<img width="240" height="25" alt="image" src="https://github.com/user-attachments/assets/b9da1af0-8b7c-4d79-9bd4-735ad952f023" />

#### CHECK FOR MISSING VALUES:

<img width="183" height="275" alt="image" src="https://github.com/user-attachments/assets/d38a40c9-f32f-428d-8477-b2cd5fbca477" />

#### CHECK FOR DUPLICATE VALUES:

<img width="172" height="452" alt="image" src="https://github.com/user-attachments/assets/97e2100d-45c0-433b-87c2-13ed8b0e6db7" />

#### DATASET STATISTICAL SUMMARY:

<img width="1255" height="283" alt="image" src="https://github.com/user-attachments/assets/4a494960-40dd-431a-84ee-a77f9c2c1b09" />

#### NORMALIZED DATASET:

<img width="612" height="474" alt="image" src="https://github.com/user-attachments/assets/ae38b855-bb28-449c-b091-38352849518f" />

#### TRAINING DATA:

<img width="356" height="147" alt="image" src="https://github.com/user-attachments/assets/33bed316-4375-4ceb-a708-937d3fdc3213" />

#### TESTING DATA:

<img width="370" height="146" alt="image" src="https://github.com/user-attachments/assets/e96af99f-72d9-4a10-b3ab-5c47fa5e2878" />

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.
