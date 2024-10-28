import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("georgia_precinct_voter_data.csv")

X = data.drop("turnout", axis=1)
y = data["turnout"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = XGBClassifier()
lgb_model = LGBMClassifier()
rf_model = RandomForestClassifier(random_state=42)

xgb_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

xgb_preds = xgb_model.predict(X_test)
lgb_preds = lgb_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

print("Model Performance on Voter Turnout Prediction\n" + "="*40)
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_preds))
print("LightGBM Accuracy:", accuracy_score(y_test, lgb_preds))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))
