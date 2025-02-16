import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingRegressor, ExtraTreesRegressor, RandomForestRegressor

# Read the data
train = pd.read_csv('train.csv')

# Preprocess the data
train_x = train.drop(columns=['Item_Outlet_Sales'])
train_y = train['Item_Outlet_Sales']

num_col = train_x.select_dtypes(include=['int64', 'float64']).columns
ohe_col = ['Item_Type']
ord_col = ['Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

outlet_size = ['Small', 'Medium', 'High']
outlet_location_type = ['Tier 1', 'Tier 2', 'Tier 3']
outlet_type = ['Supermarket Type1', 'Supermarket Type2',
               'Supermarket Type3', 'Grocery Store']

oe = OrdinalEncoder(categories=[outlet_size, outlet_location_type, outlet_type],
                    handle_unknown='use_encoded_value', unknown_value=np.nan)

ct = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_col),
        ('ohe', OneHotEncoder(sparse_output=False), ohe_col),
        ('ord', oe, ord_col)
    ], remainder='drop', n_jobs=-1)

# Model
simple_etr = Pipeline([
    ('transformer', ct),
    ('imputer', SimpleImputer(strategy='mean')),
    ('model', ExtraTreesRegressor(max_depth=7, n_jobs=-1))
])

knn_etr = Pipeline([
    ('transformer', ct),
    ('imputer', KNNImputer(n_neighbors=16)),
    ('lr', ExtraTreesRegressor(max_depth=7, n_jobs=-1))
])

knn_rfr = Pipeline([
    ('transformer', ct),
    ('imputer', KNNImputer(n_neighbors=12)),
    ('lr', RandomForestRegressor(max_depth=5, n_jobs=-1))
])

vt = VotingRegressor([
    ('simple_etr', simple_etr),
    ('knn_etr', knn_etr),
    ('knn_rfr', knn_rfr)
], n_jobs=-1)

# Fit the model
vt.fit(train_x, train_y)

# Predict and save the submission
test = pd.read_csv('test.csv')

test_serial_id = test[['Item_Identifier', 'Outlet_Identifier']]

predictions = vt.predict(test)
predictions = pd.DataFrame(predictions, columns=['Item_Outlet_Sales'])

submission = pd.concat([test_serial_id, predictions], axis=1)
submission.to_csv('submission.csv', index=False)
