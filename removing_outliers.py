import pandas as pd
df = pd.read_csv('bengaluru_house_prices.csv')
print(df)

print(df.isna().sum())

df['location'] = df['location'].ffill().bfill()

print(df.isna().sum())

df['size'] = df['size'].str.strip(' ').str[0]
df['size'] =df['size'].astype(float)
df['size'] = df['size'].fillna(df['size'].median())

df = df.drop(columns='society')

df['bath'] = df['bath'].fillna(df['bath'].median())
df['balcony'] = df['balcony'].fillna(df['balcony'].median())

df['area_type'] = df['area_type'].astype(str)
df['availability'] = df['availability'].astype(str)
df['location'] = df['location'].astype(str)

def is_float(x):
    try:
        float(x)
    except:
        return (False)
    return True
df['total_sqft'].apply(is_float)

def convert_sqft_to_num (x):
    try:
        x = str(x)
        if '-' in x:
            low,high = x.split('-')
            return(float(low)+float(high))/2
        else:
            return float(x)
    except:
        return None
    
df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)

numeric_cols = df[['size','total_sqft','bath','balcony']]

for col in numeric_cols:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df = df[(df[col]>=lower) & (df[col]<=upper)]

area_type_dummies = pd.get_dummies(df['area_type'],dtype=float)
availability_dummies = pd.get_dummies(df['availability'],dtype=float)
location_dummies = pd.get_dummies(df['location'],dtype=float)

categorical = df[['area_type','availability','location']]
categorical_dummies = pd.get_dummies(categorical,dtype=float)

x = pd.concat([location_dummies,
               df['size'],
               df['total_sqft'],
               df['bath'],
               df['balcony']],axis='columns')

y = df['price']

#Importing models/ best params/ best scores
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("\n=== REGRESSION MODELS ===\n")

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

