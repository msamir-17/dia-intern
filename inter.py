#STEP1
import pandas as pd

#STEP2
salary = pd.read_csv("https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Diabetes.csv")
salary.head()

salary.columns
#STEP3
y = salary['diabetes']
x = salary[['pregnancies', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi',
       'dpf', 'age']]

#STEP4
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=2529)

#STEP5 SELECT A MODEL
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()

#STEP6 TRAIN MODEL
model.fit(x_train,y_train)

#step7
y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score
