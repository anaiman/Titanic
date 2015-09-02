# -*- coding: utf-8 -*-
"""
Created on Tue Sep 01 14:04:06 2015

@author: anaiman
"""

import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier

# Define some mappings
sexToInt = {'female': 0, 'male': 1}
embarkedToInt = {'C': 0, 'Q' : 1, 'S' : 2}

def prepData(csvf):
    # Load training set
    df = pd.read_csv(csvf, header = 0)
    
    # All missing Embarked -> just make them embark from most common place
    if len(df.Embarked[ df.Embarked.isnull() ]) > 0:
        df.Embarked[df.Embarked.isnull()] = df.Embarked.dropna().mode().values
    
    # Make strings into ints
    df['Gender'] = df.Sex.map(sexToInt)
    df['iEmbarked'] = df.Embarked.map(embarkedToInt)
    
    # Predict ages to fill in blanks
    nGender = len(np.unique(df.Gender))
    nClass = len(np.unique(df.Pclass))
    medianAge = np.zeros((nGender, nClass))
    for i in range(0, nGender):
        for j in range(0, nClass):
            medianAge[i,j] = df[(df.Gender == i) & \
                                (df.Pclass == j+1)].Age.dropna().median()
    df['AgeFill'] = df.Age
    for i in range(0, nGender):
        for j in range(0, nClass):
            df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), \
                'AgeFill'] = medianAge[i,j]
    df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
    
    # All the missing Fares -> assume median of their respective class
    if len(df.Fare[ df.Fare.isnull() ]) > 0:
        median_fare = np.zeros(3)
        for f in range(0,3):                                              # loop 0 to 2
            median_fare[f] = df[ df.Pclass == f+1 ]['Fare'].dropna().median()
        for f in range(0,3):                                              # loop 0 to 2
            df.loc[ (df.Fare.isnull()) & (df.Pclass == f+1 ), 'Fare'] = median_fare[f]
    
    clean = df.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1)
    return df.PassengerId.values, clean.values

if __name__ == "__main__":
    ids, td = prepData('train.csv')
    ids, test = prepData('test.csv')
    
    print 'Training...'
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit( td[0::,1::], td[0::,0] )
    
    print 'Predicting...'
    output = forest.predict(test).astype(int)
    
    
    predictions_file = open("myfirstforest.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId","Survived"])
    open_file_object.writerows(zip(ids, output))
    predictions_file.close()
    print 'Done.'