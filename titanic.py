
import pandas as pd
import numpy as np
import pydotplus
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz

# Read the train and test data

# PClas = Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd
# SibSp  = # Brothers/sisters/spouses aboard the Titanic
# Parch = # parents / children aboard the Titanic
# Fare = Price of cabinet 

# [Pclass, Age, SibSp, Parch, Fare, Embarked]
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

#Cleaning train data

# "Sex" Coulumn has male/female as value. We can use LabelEncoder
# to convert these to int. male:1,female:0
lb = LabelEncoder()
train_df['Sex'] = lb.fit_transform(train_df['Sex']) 

# Dataset descriptions defines three values (C,Q,S) as port of embarkation
count_null_embarked = len(train_df['Embarked'][ train_df.Embarked.isnull() ])
lb2 = LabelEncoder()
train_df['Embarked'] = lb2.fit_transform(train_df['Embarked']) 

# Set the target column to Survived
targets = train_df.Survived

#Dropping unwanted columns. Also, removing the target column.
train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId','Survived'], axis=1) 
#Imputer is used to fill all the occurances of NaN with mean of that column.
im = Imputer()
predictors = im.fit_transform(train_df)

#Using Decision Tree Classifier
classifier=DecisionTreeClassifier()
classifier=classifier.fit(predictors,targets)


lb3 = LabelEncoder()
test_df['Sex'] = lb3.fit_transform(test_df['Sex']) #male:1, female:0

count_null_embarked = len(test_df.Embarked[ test_df.Embarked.isnull() ])

lb4 = LabelEncoder()

test_df['Embarked'] = lb4.fit_transform(test_df['Embarked']) 

test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 
im2 = Imputer()
test_predictors = im2.fit_transform(test_df)

#Making predictions on the test data
predictions=classifier.predict(test_predictors)

#Creating Output file as required by kaggle
test_data = pd.read_csv("test.csv").values

result = np.c_[test_data[:,0].astype(int), predictions.astype(int)]
df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])
df_result.to_csv('res1.csv', index=False)

# lets split the train_df and do a validation
pred_train, pred_validation, tar_train, tar_validation  =   train_test_split(predictors, targets, test_size=.4)
print("Validation test....")
dt_classifier=DecisionTreeClassifier()
dt_classifier=dt_classifier.fit(pred_train,tar_train)

validations=dt_classifier.predict(pred_validation)
print("Accuracy... "+str(sklearn.metrics.accuracy_score(tar_validation, validations)))

# #Generating decision tree in a file(image)
out = StringIO()
export_graphviz(classifier, out_file=out,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(out.getvalue())  
Image(graph.create_png())
graph.write_pdf("titanic.pdf")



