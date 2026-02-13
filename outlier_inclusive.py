#Importing file using pandas
import pandas as pd
df = pd.read_csv('bengaluru_house_prices.csv')
print(df)

#Checking columns with Null values
print(df.isna().sum())

#Filling Null values
df['location'] = df['location'].ffill().bfill()
print(df.isna().sum())

#Restructuring size column
df['size'] = df['size'].str.split(' ').str[0]
df['size'] = df['size'].astype(float)

df['size'] = df['size'].fillna(df['size'].median())
print(df.isna().sum())

#Dropping society column 
df =df.drop(columns='society')

df['bath'] = df['bath'].fillna(df['bath'].median())
df['balcony'] = df['balcony'].fillna(df['balcony'].median())

#Converting total_sqft column to float
def convert_sqft (x):
    try:
        x = str(x)
        if '-' in x:
            low,high = x.split('-')
            return (float(low)+float(high))/2
        else:
            return float(x)
    except:
        return None
df['total_sqft'] = df['total_sqft'].apply(convert_sqft)

df['total_sqft'] = df['total_sqft'].fillna(df['total_sqft'].median())

#Creating dummy variables
cols = df[['area_type','availability','location']].astype(str)
area_type_dummies = pd.get_dummies(df['area_type'],dtype=float)
availability_dummies = pd.get_dummies(df['availability'],dtype=float)
location_dummies = pd.get_dummies(df['location'],dtype=float)

categorical = cols
numerical = df[['size','total_sqft','bath','balcony']]
dummies = pd.get_dummies(categorical,dtype=float)

x = pd.concat([df['size'],
               df['bath'],
               df['total_sqft'],
               location_dummies],axis='columns')

y = df['price']

#Importing models/ best params/ best scores
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Hyperparameter tuning using GridSearchCV
# Linear Regression
reg_pipe = Pipeline([
    ('scaler',StandardScaler()),
    ('reg',LinearRegression())
])

reg_grid = GridSearchCV(
    reg_pipe,
    {},
    cv=5,
    scoring='r2',
    n_jobs=-1
)
reg_grid.fit(x_train,y_train)

print("Linear Regression best params:",reg_grid.best_params_)
print('Best CV R2:',reg_grid.best_score_)  

best_reg = reg_grid.best_estimator_
print('Train R2:',best_reg.score(x_train,y_train))
print('Test R2:',best_reg.score(x_test,y_test))

# SVR (scaled)
svr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR())
])
svr_grid = GridSearchCV(
    svr_pipe,
    {
        "svr__kernel": ["linear", "rbf"],
        "svr__C": [0.01, 0.1, 1, 10,100],
        "svr__gamma": ["scale"]
    },
    cv=5,
    scoring="r2",
    n_jobs=-1
)
svr_grid.fit(x_train, y_train)
best_svr = svr_grid.best_estimator_
print("SVR best params:", svr_grid.best_params_)
print('Train R2:',best_svr.score(x_train,y_train))
print('Test R2:',best_svr.score(x_test,y_test))

# Random Forest Regressor
rfr_grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    {
        "n_estimators": [100,300],
        "max_depth": [None, 10],
        "min_samples_split": [3,6],
        "min_samples_leaf": [2,4],
        "max_features": ['sqrt',0.5]
    },
    cv=5,
    scoring="r2",
    n_jobs=-1
)
rfr_grid.fit(x_train, y_train)
best_rfr = rfr_grid.best_estimator_
print("RF best params:", rfr_grid.best_params_)
print('Train R2:',best_rfr.score(x_train,y_train))
print('Test R2:',best_rfr.score(x_test,y_test))


