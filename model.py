#import necessary libraries
import pickle

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

#create Dataset
df=pd.read_csv('crop_recommendation.csv')

#Extract X and y
X=df[['N','P','K','temperature','humidity','ph','rainfall']]
y=df['label']

# create Model
model=DecisionTreeClassifier()

# train the model
model.fit(X,y)

# save the model
with open('crop_model.pkl','wb') as f:
   pickle.dump(model,f)
#make prediction
y_pred=model.predict([[90,42,43,20.87974371,82.00274423,6.502985292000001,202.9355362]])
print(y_pred)