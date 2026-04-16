import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

from sklearn.metrics import mean_absolute_percentage_error

try:
    # Load dataset
    print("Loading the dataset")
    file_path = "C:/Users/varra/OneDrive/Documents/ML Proj/Insurance CaseStudy/insurancedata.csv"
    data = pd.read_csv(file_path)

    # Handling Missing Values
    print("Data preprocessing")
    for col in data.columns:
        if data[col].isnull().any():
            if data[col].dtype == 'object':
                # Fill missing categorical values with mode (most frequent)
                data[col].fillna(data[col].mode()[0], inplace=True)
            else:
                # Fill missing numeric values with median
                data[col].fillna(data[col].median(), inplace=True)

    # Encoding Categorical Variables
    categorical_cols = ['district', 'region', 'spendtype']
    if any(col in data.columns for col in categorical_cols):
        encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
        encoded_data = pd.DataFrame(encoder.fit_transform(data[categorical_cols]).toarray(),
                                    columns=encoder.get_feature_names_out(categorical_cols))
        data = data.drop(columns=categorical_cols).reset_index(drop=True)
        data = pd.concat([data, encoded_data], axis=1)

    # Features and Target
    if 'avgcancerspend' in data.columns:
        X = data.drop(columns=['avgcancerspend'])  # Exclude target column
        y = data['avgcancerspend']
    else:
        raise ValueError("Target column 'avgcancerspend' is missing from the dataset.")

    # Splitting dataset into training and testing sets
    print("Splitting the dataset")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #Training the model
    print("Training the model")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Making Predictions
    print("Making predictions")
    y_pred = model.predict(X_test)

    # Model Performance Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    accuracy = 1 - mape

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
    print(f"Accuracy (based on MAPE): {accuracy:.2f}")

    # Saving Predictions and Performance Metrics
    test_predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    test_predictions.to_csv('cancer_spend_predictions.csv', index=False)

    performance_df = pd.DataFrame({'Metric': ['MSE', 'R-squared', 'Accuracy'], 
                                   'Value': [mse, r2, accuracy]})
    performance_df.to_csv("performance2.csv", index=False)

    # Data Visualization: Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel("Actual Avg Cancer Spend")
    plt.ylabel("Predicted Avg Cancer Spend")
    plt.title("Actual vs Predicted Avg Cancer Spend")
    plt.show()

    print("Predictions and performance saved successfully!")
    print("Script Executed Successfully!")

except FileNotFoundError:
    print(f"Error: The dataset file '{file_path}' was not found.")
except ValueError as ve:
    print(f"Value Error: {ve}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")





