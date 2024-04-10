from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pylab as plt
import sys
from dmba import classificationSummary

C5nb = pd.read_csv('Analysis.csv')
C5nb = C5nb.drop(columns=['EDUC','GENDER','MARRIAGE','OWNSTATE','RELIG1','SELFBORN','Volunteer'])

print(C5nb.head())

print(C5nb.dtypes)

predictors = ['Education_Level','Male','Marriage_Status','Own_State','Christian','BORN_USA']
y = C5nb['VOLUNT1']
X = C5nb[predictors]

X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size = .6, random_state = 256)

volunt_nb = MultinomialNB(alpha=.01)
volunt_nb.fit(X_train, y_train)

predProb_train = volunt_nb.predict_proba(X_train)
predProb_valid = volunt_nb.predict_proba(X_valid)

y_valid_pred = volunt_nb.predict_proba(X_valid)
y_train_pred = volunt_nb.predict(X_train)

train_df, valid_df = train_test_split(C5nb, test_size=0.50, random_state=1)
pd.set_option('display.precision', 4) 
print(train_df['VOLUNT1'].value_counts() / len(train_df))
print()



# Marriage_Status: 0-Never Married 1- Married 2- Divorced 3 - Widowed

# Own_State: 0-Poor 1-Fair 2-Good 3-Very Good 4- Excellent

# Education_Level: 0-HighSchool 1-Bachelors 2-Masters 3-PHD



for predictor in predictors:
    df = train_df[['VOLUNT1', predictor]]
    freqTable = df.pivot_table(index='VOLUNT1', columns=predictor, aggfunc=len)
    propTable = freqTable.apply(lambda x: x / sum(x), axis=1)
    print(propTable)
    print()
    
pd.reset_option('display.precision')

#Yes, bachelors degree, female,married,verygood, Christian, Born in USA
likely_volun = .5832 * .5259 * .6519 * .3444 * .8259 * .8519
notlike_volun = .4168*.3109*.4663*.5389*.3161*.7720*.7409

print((likely_volun/(likely_volun+notlike_volun))*100)
print((notlike_volun/(likely_volun+notlike_volun))*100)