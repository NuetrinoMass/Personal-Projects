#Import all packages
import pandas as pd
import warnings
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as sm_api
import matplotlib.pylab as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dmba import plotDecisionTree, classificationSummary, regressionSummary


#import csv file
df = pd.read_csv('TelcoCustomerChurn.csv')

#creating dummy variables for gender, tenure, Contract, Partner,Dependents

cls_df = df[['SeniorCitizen','gender','tenure','Partner','Dependents'
             ,'Contract','TechSupport','Churn']]
cls_df= pd.get_dummies(cls_df, columns = ['Partner','Dependents'
             ,'Contract','TechSupport'])
cls_df[['Churn']] = cls_df[['Churn']].replace({'Yes':1,'No':0})
cls_df['gender'] = cls_df['gender'].replace({'Male':0,'Female':1})
cls_df.head()

##correlation becomes hard to visualize with so many aggregated components
corr = cls_df.corr()
#sets correlation so that all values are correlations to Churn 
corr['Churn'].sort_values(ascending = False).plot(kind ='bar',color= 'g', 
                                                  title = 'DA1: Correlation of Variables to Churn Bar')
plt.show()

ax = plt.axes()
sns.heatmap(corr,annot = True, linewidth = .5, cmap = 'PiYG', ax = ax)
ax.set_title('DA1a: Correlation Matrix with annotation')
plt.show()

cls_df.boxplot(column = ['tenure'],by='Churn').set_title('DA2b: Tenure Grouped by Churn')
plt.show()

sns.barplot(x='Churn',y='tenure',hue='gender',data = cls_df).set_title(
    'DA2c: Break down of Churn as related to Tenure and Gender ')
plt.show()

sns.barplot(x='Churn',y = 'Contract_Month-to-month',hue='gender',data= cls_df).set_title(
    'DA2d: Break down of Churn as related to Month to Month and Gender')
plt.show()

sns.barplot(x='Churn',y='Contract_Two year', hue = 'gender', data = cls_df).set_title(
    'DA2e: Break down of Churn as related to Two year Contract and Gender')
plt.show()

#dropping any possible null values as they will mess up the Learning Algorithms
cls_df = cls_df.dropna()
#setting the variable we want predicted to be Churn
y = cls_df['Churn']
#setting all other variables besides Churn to be indepedent variables. 
X= cls_df.drop(columns=['Churn'])

#Splitting the dataset into 70 percent training data and 30 percent testing
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size= .3, random_state = 105)


###Decision Tree
print('\n\n Decision Tree')
x = 0
warnings.filterwarnings('ignore')
# using i as indicator of the depth to see at which depth the highest precision can be achieved
for i in range(1,100):
    mTree = DecisionTreeClassifier(criterion='entropy', random_state=45, max_depth=i)
    mTree.fit(X_train,y_train)
    y_hat = mTree.predict(X_test)
    accuracy = accuracy_score(y_test,y_hat)
    if accuracy >x:
        x = round(accuracy,2)
        depth = i
        precision = round(precision_score(y_test,y_hat),2)
print(f"With depth of {depth} our accuracy is:{x}")
print(f"With depth of {depth} our precision is: {precision}")

#Setting the sct or Small Class Tree for visualizeation and use in a confusion Matrix
sct = DecisionTreeClassifier(max_depth = 5,min_impurity_decrease = .01)
#fitting the data into the sct
sct.fit(X_train,y_train)

print('Confusion Matrix for Train data')
# allows us to see the quanity of predictions missed by the Tree model for the train data set 
classificationSummary(y_train,sct.predict(X_train))

print('\n\nConfusion Matrix for Test data')
# Allows us to see the True and false positives/negatives. A visual representation in a way of how our model is preforming. 
classificationSummary(y_test,sct.predict(X_test))

#to visualize the tree and the criteria in which it breaks it down. 
print(plotDecisionTree(sct, feature_names = X_train.columns))

#KnearestNeighbors
print('\n\n KnearestNeighbors')

nn = 10
for i in range(7, nn+1):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    y_hat= knn.predict(X_test)
    print(f' k = {i} Accuracy: {round(accuracy_score(y_test,y_hat),2)}')
    print(f' k = {i} Precision {round(precision_score(y_test,y_hat),2)}')
    print()