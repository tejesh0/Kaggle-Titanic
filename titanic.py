
m sklearn.ensemble import RandomForestClassifier 
from sklearn.cross_validation import cross_val_score
import numpy as np
from sklearn.preprocessing import Imputer
import pandas 
path = '/home/supertramp/Downloads/titanic_data.csv'
path1 = '/home/supertramp/Downloads/test.csv'

tdf  = pandas.read_csv(path1)

mydf = pandas.read_csv(path)


mydf.Fare =mydf.Fare.map(lambda x: np.nan if x==0 else x)


classmeans = mydf.pivot_table('Fare',rows='Pclass',aggfunc='mean')

mydf.Fare = mydf[['Fare','Pclass']].apply(lambda x: classmeans[x['Pclass']] if pandas.isnull(x['Fare']) else x['Fare'],axis = 1)


meanAge = np.median(mydf.Age)
mydf.Age = mydf.Age.fillna(meanAge)




mydf['Gender'] = mydf['Sex'].map({'male' : 1 , 'female' : 0}).astype(int)

print mydf['Gender']

print mydf[ mydf.Embarked != 'S' ][['Survived']]

mydf.Embarked = mydf.Embarked.map(lambda x : 1 if x=='S' else 2)


#mydf['01Embark'] = mydf['Embarked'].map({'C' : 1 ,'S' : 2 ,'Q': 3}).astype(int)
#print mydf['01Embark']

mydf = mydf.drop(['Sex','Cabin','Name','Ticket'],axis = 1)


print mydf.dtypes[mydf.dtypes.map(lambda x :  x!='object')]


train_data = mydf.values

tdf = tdf.drop(['Sex','Cabin','Name','Ticket','Embarked'],axis=1)

print '--------------------'
print tdf.dtypes[tdf.dtypes.map(lambda x :x == 'object')]
test = tdf.values
# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data[0::,1::],train_data[0::,0])

columns = ['Pclass','Gender']
features = mydf[list(columns)].values
labels = mydf['Survived'].values

test_df = pandas.read_csv(path1)
test_df['Gender'] = test_df['Sex'].map({'male' : 1 , 'female' : 0}).astype(int)
test_df['Embarked'] = test_df['Embarked'].map({'S' : 1,'C' :2,'Q' :0 }).astype(int)
agemean = np.median(test_df.Age)
test_df.Age = test_df.Age.fillna(agemean)
forest = forest.fit(features, labels)
forest.predict(test_df[columns].values)
et_score = cross_val_score(forest, features, labels, n_jobs=-1).mean()

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(features)

 
forest.fit(features, labels)
predictions =  forest.predict(imp.transform(test_df[columns].values))
test_df["Survived"] = pandas.Series(predictions)

# Take the same decision trees and run it on the test data

test_df.to_csv("/home/supertramp/Desktop/finalPrediction.csv", cols=['PassengerId','Survived'], index=False)

