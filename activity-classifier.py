import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

X = df[metrics]
y = df['Category']

# Encode the categories into numerical labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

unclear_label = len(label_encoder.classes_)  # Assign a label not used by other classes

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Add 'Unclear' to the label encoder
label_encoder.classes_ = np.append(label_encoder.classes_, 'Unclear')


# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

xgb = XGBClassifier(objective='multi:softmax', num_class=len(np.unique(y_train)), random_state=42)

# Define the parameter grid for sequential grid search
param_grid_seq = {
 'max_depth':range(3,10,1),
 'min_child_weight':range(1,6,1),
 'subsample':[i/10.0 for i in range(6,11)],
 'colsample_bytree':[i/10.0 for i in range(6,11)],
 'learning_rate':[0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9],
 'gamma':[i/10.0 for i in range(0,5)],
 'n_estimators':range(0, 1100, 100)
}

# Sequential grid search
for param_name in param_grid_seq.keys():
    # Define the parameter grid for the current iteration
    param_grid = {param_name: param_grid_seq[param_name]}

    # Initialize GridSearchCV with one parameter
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Extract the best parameter value
    best_param_value = grid_search.best_params_[param_name]
    print("Best", param_name, ":", best_param_value)
    print("Best score: ", grid_search.best_score_)

    # Update the parameter grid for the next iteration
    param_grid_seq[param_name] = [best_param_value]

# After sequential grid search, print the best parameters found
print("Best Parameters:", param_grid_seq)
