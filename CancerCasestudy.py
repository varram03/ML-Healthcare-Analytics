try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.preprocessing import StandardScaler

    # Load the data
    data = pd.read_csv('wisc_bc_data.csv')
    print("Columns in dataset:", data.columns)
    print("First few rows:\n", data.head())
    print("Load the data")

    # Convert target to binary
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    print("Unique values in 'diagnosis':", data['diagnosis'].unique())

    # Handle missing values
    data.fillna(data.median(), inplace=True)
    print("Preprocess the data")

    # Bar plots
    sns.countplot(x='diagnosis', data=data)
    plt.show()
    print("Visualizing the data")

    # Split dataset
    X = data.drop(columns=['id', 'diagnosis'])
    y = data['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print("Split the dataset")

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train and Evaluate SVM Model
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    print("Training and Evaluating the model")

    accuracy=accuracy_score(y_test,y_pred)
    report=classification_report(y_test,y_pred,output_dict=True)

    performance_df=pd.DataFrame(report).transpose()
    performance_df["accuracy"]=accuracy
    performance_df.to_csv("performance.csv",index=True)
    test_predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    test_predictions.to_csv('cancer_predictions.csv', index=False)

    print('Predictions:\n',y_pred)
    print("Actual Values:\n",y_test.values)
    print(f"Accuracy: {accuracy:.2f}")
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)
    print(classification_report(y_test,y_pred))
    print("Predicitons and performance metrics saved successfully!")

except Exception as e:
    print(f"Error encountered: {e}")



