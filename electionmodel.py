import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Election data (2008-2016 for prediction of 2018 election)
election_data = {
    'County': ['Montgomery', 'Hays', 'Hidalgo', 'Tarrant', 'Bell', 'Travis', 'Dallas', 'El Paso', 'Williamson', 'Cameron', 'Bexar', 'Brazoria', 'Lubbock', 'Collin', 'Denton'],
    
    # 2008 Presidential Election Data
    'Obama_2008_Percent': [26.00, 56.00, 68.00, 45.00, 37.00, 64.00, 57.00, 66.00, 43.00, 64.00, 53.00, 37.00, 28.00, 38.00, 37.00],
    'McCain_2008_Percent': [73.00, 43.00, 31.00, 55.00, 62.00, 35.00, 42.00, 33.00, 56.00, 35.00, 46.00, 62.00, 71.00, 61.00, 62.00],
    
    # 2012 Presidential Election Data
    'Obama_2012_Percent': [24.00, 55.00, 70.00, 43.00, 36.00, 60.00, 57.00, 68.00, 42.00, 63.00, 52.00, 35.00, 27.00, 33.00, 34.00],
    'Romney_2012_Percent': [75.00, 44.00, 29.00, 57.00, 63.00, 38.00, 42.00, 31.00, 58.00, 36.00, 47.00, 64.00, 72.00, 66.00, 65.00],

    # 2016 Presidential Election Data
    'Trump_2016_Percent': [73.00, 46.87, 27.89, 52.23, 54.33, 33.00, 38.00, 25.00, 51.00, 32.00, 40.00, 58.00, 66.00, 56.00, 60.00],
    'Clinton_2016_Percent': [22.26, 46.04, 68.12, 43.54, 39.50, 61.00, 60.00, 69.00, 43.00, 65.00, 54.00, 38.00, 30.00, 39.00, 34.00],
    
    # 2018 Senate Election (Cruz vs. O'Rourke) - Actual results for validation
    'Cruz_2018_Percent': [72.28, 45.31, 30.64, 50.92, 56.73, 39.00, 38.00, 27.00, 54.00, 32.00, 40.00, 60.00, 67.00, 57.00, 60.00],
    'O_Rourke_2018_Percent': [26.97, 53.40, 68.81, 47.65, 42.81, 60.00, 61.00, 71.00, 45.00, 66.00, 58.00, 38.00, 32.00, 41.00, 36.00],
}

# Create DataFrame from election data
df_elections = pd.DataFrame(election_data)

# Feature set from 2008-2016 data
df_model = pd.DataFrame({
    'County': ['Montgomery', 'Hays', 'Hidalgo', 'Tarrant', 'Bell', 'Travis', 'Dallas', 'El Paso', 'Williamson', 'Cameron', 'Bexar', 'Brazoria', 'Lubbock', 'Collin', 'Denton'],
    'Obama_2008_Percent': [26.00, 56.00, 68.00, 45.00, 37.00, 64.00, 57.00, 66.00, 43.00, 64.00, 53.00, 37.00, 28.00, 38.00, 37.00],
    'McCain_2008_Percent': [73.00, 43.00, 31.00, 55.00, 62.00, 35.00, 42.00, 33.00, 56.00, 35.00, 46.00, 62.00, 71.00, 61.00, 62.00],
    'Obama_2012_Percent': [24.00, 55.00, 70.00, 43.00, 36.00, 60.00, 57.00, 68.00, 42.00, 63.00, 52.00, 35.00, 27.00, 33.00, 34.00],
    'Romney_2012_Percent': [75.00, 44.00, 29.00, 57.00, 63.00, 38.00, 42.00, 31.00, 58.00, 36.00, 47.00, 64.00, 72.00, 66.00, 65.00],
    'Trump_2016_Percent': [73.00, 46.87, 27.89, 52.23, 54.33, 33.00, 38.00, 25.00, 51.00, 32.00, 40.00, 58.00, 66.00, 56.00, 60.00],
    'Clinton_2016_Percent': [22.26, 46.04, 68.12, 43.54, 39.50, 61.00, 60.00, 69.00, 43.00, 65.00, 54.00, 38.00, 30.00, 39.00, 34.00],
})

# Adding campaign issue importance based on county political leaning and location
campaign_issues = {
    'County': ['Montgomery', 'Hays', 'Hidalgo', 'Tarrant', 'Bell', 'Travis', 'Dallas', 'El Paso', 'Williamson', 'Cameron', 'Bexar', 'Brazoria', 'Lubbock', 'Collin', 'Denton', 'Harris', 'Fort Bend', 'Smith'],
    
    # Importance of each campaign issue (scale 0-1, 1 being most important)
    'Economy_Importance': [0.95, 0.7, 0.55, 0.85, 0.75, 0.4, 0.5, 0.6, 0.7, 0.6, 0.5, 0.9, 0.95, 0.8, 0.8, 0.4, 0.47, 0.9],
    'Immigration_Importance': [0.9, 0.75, 0.95, 0.75, 0.8, 0.3, 0.4, 0.9, 0.7, 0.9, 0.55, 0.85, 0.9, 0.65, 0.7, 0.5, 0.47, 0.9],
    'Healthcare_Importance': [0.3, 0.6, 0.7, 0.4, 0.5, 0.85, 0.9, 0.8, 0.55, 0.7, 0.85, 0.4, 0.35, 0.6, 0.6, 0.8, 0.57, 0.31],
    'Abortion_Importance': [0.3, 0.6, 0.7, 0.5, 0.5, 0.85, 0.9, 0.8, 0.55, 0.75, 0.75, 0.4, 0.35, 0.6, 0.6, 0.8, 0.62, 0.24],
    'Education_Importance': [0.5, 0.65, 0.6, 0.6, 0.55, 0.8, 0.8, 0.7, 0.65, 0.7, 0.75, 0.6, 0.5, 0.65, 0.65, 0.6, 0.47, 0.9]
}

# Polling data adjustment to slightly favor Cruz
general_polling_data = {
    'County': ['Montgomery', 'Hays', 'Hidalgo', 'Tarrant', 'Bell', 'Travis', 'Dallas', 'El Paso', 'Williamson', 'Cameron', 'Bexar', 'Brazoria', 'Lubbock', 'Collin', 'Denton', 'Harris', 'Fort Bend', 'Smith'],
    'Cruz_Poll_Average': [72.0, 47.0, 44.0, 51.5, 53.0, 38.0, 38.0, 39.0, 46.0, 47.0, 42.0, 57.0, 56.0, 54.0, 56.0, 43.0, 47.0, 68.0],  # Slight adjustment to Cruz's numbers
    'ORourke_Poll_Average': [26.0, 51.0, 53.0, 46.0, 44.0, 57.0, 60.0, 60.0, 51.0, 51.0, 56.0, 40.0, 42.0, 43.0, 42.0, 55.0, 52.0, 30.0],
    'Undecided_Voters': [2.0, 2.0, 3.0, 3.5, 3.5, 2.0, 2.0, 1.0, 3.0, 3.5, 4.0, 2.0, 3.0, 3.0, 1.5, 1.0, 1.0, 1.5]
}

# Create DataFrame from polling data and merge into the model DataFrame
df_polling = pd.DataFrame(general_polling_data)
df_model['Cruz_Poll_Average'] = df_polling['Cruz_Poll_Average']
df_model['ORourke_Poll_Average'] = df_polling['ORourke_Poll_Average']
df_model['Undecided_Voters'] = df_polling['Undecided_Voters']

# Merge campaign issues into the model DataFrame
df_issues = pd.DataFrame(campaign_issues)
df_model['Economy_Importance'] = df_issues['Economy_Importance']
df_model['Immigration_Importance'] = df_issues['Immigration_Importance']
df_model['Healthcare_Importance'] = df_issues['Healthcare_Importance']
df_model['Abortion_Importance'] = df_issues['Abortion_Importance']
df_model['Education_Importance'] = df_issues['Education_Importance']


# Proceed with the rest of the model feature set
features = df_model[['Obama_2008_Percent', 'McCain_2008_Percent', 'Obama_2012_Percent', 'Romney_2012_Percent', 
                     'Trump_2016_Percent', 'Clinton_2016_Percent', 'Cruz_Poll_Average', 'ORourke_Poll_Average', 
                     'Undecided_Voters', 'Economy_Importance', 'Immigration_Importance', 'Healthcare_Importance', 
                     'Abortion_Importance', 'Education_Importance']]

# Target: 2018 Senate Results
target_cruz = df_elections['Cruz_2018_Percent']
target_orourke = df_elections['O_Rourke_2018_Percent']

# Scaling features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Train-Test Split for Cruz's prediction
X_train_cruz, X_test_cruz, y_train_cruz, y_test_cruz = train_test_split(scaled_features, target_cruz, test_size=0.3, random_state=42)

# Train-Test Split for O'Rourke's prediction
X_train_orourke, X_test_orourke, y_train_orourke, y_test_orourke = train_test_split(scaled_features, target_orourke, test_size=0.3, random_state=42)

# Model 1: Random Forest Regressor for Cruz
rf_regressor_cruz = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor_cruz.fit(X_train_cruz, y_train_cruz)
y_pred_cruz_rf = rf_regressor_cruz.predict(X_test_cruz)

# Model 2: Gradient Boosting Regressor for Cruz
gb_regressor_cruz = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_regressor_cruz.fit(X_train_cruz, y_train_cruz)
y_pred_cruz_gb = gb_regressor_cruz.predict(X_test_cruz)

# Hybrid Model for Cruz (average of Random Forest and Gradient Boosting)
final_prediction_cruz = (y_pred_cruz_rf + y_pred_cruz_gb) / 2

# Model 1: Random Forest Regressor for O'Rourke
rf_regressor_orourke = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor_orourke.fit(X_train_orourke, y_train_orourke)
y_pred_orourke_rf = rf_regressor_orourke.predict(X_test_orourke)

# Model 2: Gradient Boosting Regressor for O'Rourke
gb_regressor_orourke = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_regressor_orourke.fit(X_train_orourke, y_train_orourke)
y_pred_orourke_gb = gb_regressor_orourke.predict(X_test_orourke)

# Hybrid Model for O'Rourke (average of Random Forest and Gradient Boosting)
final_prediction_orourke = (y_pred_orourke_rf + y_pred_orourke_gb) / 2

# Evaluate the models for Cruz
mse_cruz_rf = mean_squared_error(y_test_cruz, y_pred_cruz_rf)
mse_cruz_gb = mean_squared_error(y_test_cruz, y_pred_cruz_gb)
mse_cruz_hybrid = mean_squared_error(y_test_cruz, final_prediction_cruz)

r2_cruz_rf = r2_score(y_test_cruz, y_pred_cruz_rf)
r2_cruz_gb = r2_score(y_test_cruz, y_pred_cruz_gb)
r2_cruz_hybrid = r2_score(y_test_cruz, final_prediction_cruz)

# Evaluate the models for O'Rourke
mse_orourke_rf = mean_squared_error(y_test_orourke, y_pred_orourke_rf)
mse_orourke_gb = mean_squared_error(y_test_orourke, y_pred_orourke_gb)
mse_orourke_hybrid = mean_squared_error(y_test_orourke, final_prediction_orourke)

r2_orourke_rf = r2_score(y_test_orourke, y_pred_orourke_rf)
r2_orourke_gb = r2_score(y_test_orourke, y_pred_orourke_gb)
r2_orourke_hybrid = r2_score(y_test_orourke, final_prediction_orourke)

# Print results for Cruz
print(f"Cruz RF Model - Mean Squared Error: {mse_cruz_rf}, R^2 Score: {r2_cruz_rf}")
print(f"Cruz GB Model - Mean Squared Error: {mse_cruz_gb}, R^2 Score: {r2_cruz_gb}")
print(f"Cruz Hybrid Model - Mean Squared Error: {mse_cruz_hybrid}, R^2 Score: {r2_cruz_hybrid}")

# Print results for O'Rourke
print(f"O'Rourke RF Model - Mean Squared Error: {mse_orourke_rf}, R^2 Score: {r2_orourke_rf}")
print(f"O'Rourke GB Model - Mean Squared Error: {mse_orourke_gb}, R^2 Score: {r2_orourke_gb}")
print(f"O'Rourke Hybrid Model - Mean Squared Error: {mse_orourke_hybrid}, R^2 Score: {r2_orourke_hybrid}")

# Final comparison of predicted vs actual results for Cruz
predicted_vs_actual_cruz = pd.DataFrame({
    'Actual_Cruz_2018_Percent': y_test_cruz,
    'Predicted_Cruz_2018_Percent': final_prediction_cruz
})

# Final comparison of predicted vs actual results for O'Rourke
# Final comparison of predicted vs actual results for O'Rourke
predicted_vs_actual_orourke = pd.DataFrame({
    'Actual_O_Rourke_2018_Percent': y_test_orourke,
    'Predicted_O_Rourke_2018_Percent': final_prediction_orourke
})

# Output results
print("\nPredicted vs Actual Results for Cruz:")
print(predicted_vs_actual_cruz)

print("\nPredicted vs Actual Results for O'Rourke:")
print(predicted_vs_actual_orourke)

# Statewide prediction (average of counties)
predicted_cruz_statewide = predicted_vs_actual_cruz['Predicted_Cruz_2018_Percent'].mean()
predicted_orourke_statewide = predicted_vs_actual_orourke['Predicted_O_Rourke_2018_Percent'].mean()

print(f"\nPredicted Statewide Cruz Percentage: {predicted_cruz_statewide:.2f}%")
print("Actual Statewide Cruz Percentage: 50.9%")

print(f"\nPredicted Statewide O'Rourke Percentage: {predicted_orourke_statewide:.2f}%")
print("Actual Statewide O'Rourke Percentage: 48.3%")
leftover = 100-predicted_cruz_statewide-predicted_orourke_statewide
print(f"Leftover Vote Libertarian/Green/None or Undecided Voters:  {leftover:.2f}%")

# Final prediction errors
cruz_error = abs(predicted_cruz_statewide - 50.9)
orourke_error = abs(predicted_orourke_statewide - 48.3)

print(f"\nCruz Prediction Error: {cruz_error:.2f}%")
print(f"O'Rourke Prediction Error: {orourke_error:.2f}%")

