# Predicting the 2018 Texas Senate Election: A Machine Learning Approach

This project implements a machine learning framework to predict the results of the 2018 Texas Senate election between Senator Ted Cruz and Beto O'Rourke. The model leverages historical election data from 2008–2016, polling data, and issue importance to accurately forecast vote shares in the 2018 race using Random Forest and Gradient Boosting algorithms.

## Dataset

The dataset includes the following:
- **Election data (2008-2016)**: Percentages of votes for major candidates in the 2008, 2012, and 2016 U.S. Presidential elections at the county level.
- **2018 Election Results**: Actual vote percentages for Ted Cruz and Beto O'Rourke in the 2018 Senate election.
- **Polling Data**: Polling averages and projections of undecided voters by county in 2018.
- **Campaign Issues**: The importance of key issues like the economy, immigration, and healthcare by county.

## Machine Learning Models

The following machine learning models are used in this project:
1. **Random Forest Regressor**: A decision tree-based ensemble learning method used for regression tasks.
2. **Gradient Boosting Regressor**: A boosting algorithm that builds sequential models to correct errors in previous models.
3. **Hybrid Model**: The average prediction from both the Random Forest and Gradient Boosting models is taken to improve accuracy.

## Steps

1. **Data Preprocessing**: 
   - Election results and polling data are standardized using `StandardScaler`.
   - The issue importance values for each county are merged into the dataset.

2. **Model Training**: 
   - A train-test split of 70-30 is used to separate the dataset into training and testing sets.
   - Both the Random Forest and Gradient Boosting models are trained on the scaled features.
   - The hybrid model combines the results from both models.

3. **Prediction and Evaluation**: 
   - Mean Squared Error (MSE) and R² scores are used to evaluate the performance of the models.
   - Final predictions for Cruz and O'Rourke are compared to the actual 2018 results.

## Results

The hybrid model predicted the statewide vote shares for Ted Cruz and Beto O'Rourke with high accuracy:
- **Predicted Statewide Cruz Vote Share**: ~50.01%
- **Actual Statewide Cruz Vote Share**: 50.9%
- **Predicted Statewide O'Rourke Vote Share**: ~48.31%
- **Actual Statewide O'Rourke Vote Share**: 48.3%

The mean squared error and R² scores for both models are calculated, and the hybrid model yields the most balanced and accurate results.

## Running the Code

To run the project locally:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/2018TexasSenatePrediction.git
