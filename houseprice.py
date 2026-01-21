import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle

df = pd.read_csv("Bengaluru_House_Data.csv")
 


model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=4,
    n_jobs=-1,
    random_state=42
)

# df['size'].dtype
df.rename(columns={'size': 'BHK'},inplace=True)

df['location'].fillna(df['location'].mode()[0],inplace=True)

df.drop('society', axis=1, inplace=True)

df['BHK']= df['BHK'].str.extract('(\d+)').astype(float)

def convert_sqft(x):
    try:
        if '-' in x:
            a, b = x.split('-')
            return (float(a) + float(b)) / 2
        return float(x)
    except:
        return np.nan

df['total_sqft'] = df['total_sqft'].astype(str).apply(convert_sqft)
df['BHK'].fillna(df['BHK'].median(),inplace=True)
  
df['bath'].fillna(df['bath'].median(),inplace=True)

df['balcony'].fillna(df['balcony'].median(),inplace=True)

df['balcony'] = df['balcony'].replace(0, 1)

# df.isnull().sum()

# df.duplicated().sum()

df.drop_duplicates(inplace=True)

# (df==0).sum()

df = pd.get_dummies(df,columns=['area_type','availability','location'],drop_first=True)

X = df.drop('price',axis=1)

y = df['price']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
scale = StandardScaler()

num_cols = ['total_sqft', 'bath', 'balcony']

X_train[num_cols] = scale.fit_transform(X_train[num_cols])
X_test[num_cols] = scale.transform(X_test[num_cols])

model.fit(X_train,y_train)


y_pre = model.predict(X_test)
r2 = r2_score(y_test,y_pre)

 

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring='r2'
)


new_df = pd.DataFrame([{
    'area_type': 'Plot Area',
    'availability': 'Ready To Move',
    'BHK': 4,
    'location': 'Uttarahalli',
     'total_sqft': 1530,
     'bath': 2,
     'balcony': 1,
}])

new_df = pd.get_dummies(new_df,columns=['area_type','availability','location'])

new_df = new_df.reindex(columns=X.columns,fill_value=0)

new_df[num_cols] = scale.transform(new_df[num_cols])

print("CV mean score", cv_scores.mean())

print("R2 score is ", r2)
pre = model.predict(new_df)

print("price of house is --->", pre)
with open("model.pkl","wb") as file:
    pickle.dump(model,file)
print("model saved successfully")