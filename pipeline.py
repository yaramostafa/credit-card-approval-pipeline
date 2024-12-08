import pandas as pd
import numpy as np
# Data preprocessing libraries
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
# Models libraries
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split # Data splitting
# Evaluation libraries
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("credit_card_train.csv")

X = df.drop(columns=['Credit_Card_Issuing'])
y = df['Credit_Card_Issuing'] # our target 0: denied, 1: approved

categ_cols = ['Gender', 'Own_Car', 'Own_Housing']

preprocessor = ColumnTransformer(transformers=[('df', OneHotEncoder(), categ_cols)],
                                 remainder='passthrough')

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X,y_encoded, test_size=0.2, random_state=42)

log_reg = LogisticRegression()

def train_and_evaluate(model_name, model_instance):
    # the pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('standardization', StandardScaler()),
        ('ClassifierModel', model_instance)
    ])
    
    print("Training ",model_name)
    
    # Train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Convert the predictions back to 'Approved' and 'Denied' if needed
    # y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    # Evaluate performance on the entire test set
    accuracy = accuracy_score(y_test, y_pred)
    print("\nOverall Model Performance:")
    print(f"Accuracy for {model_name}: {accuracy * 100:.2f}%")
    
    # Evaluate the model's performance
    print("Classification Report:")
    print(classification_report(y_test, y_pred))  # Using numeric labels
    print("Predicted labels:", y_pred[:10]) 

    # Bias/Fairness Evaluation: Check classification reports for males and females
    male_indices = (X_test['Gender'] == 'Male').values  # Adjust as per your data format
    female_indices = (X_test['Gender'] == 'Female').values

    y_pred_male = model.predict(X_test[male_indices])
    y_true_male = y_test[male_indices]
    y_pred_female = model.predict(X_test[female_indices])
    y_true_female = y_test[female_indices]

    print("\nBias/Fairness Evaluation:")
    print(f"Male Classification Report for {model_name}:")
    print(classification_report(y_true_male, y_pred_male))
    print(f"Female Classification Report for {model_name}:")
    print(classification_report(y_true_female, y_pred_female))

    # Variance: Compare training and test performance
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print("\nVariance Check:")
    print(f"Training Accuracy for {model_name}: {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy for {model_name}: {accuracy * 100:.2f}%")

    # Get feature importances from the model (if supported by model used)
    if hasattr(model.named_steps['ClassifierModel'], 'feature_importances_'):
        importances = model.named_steps['ClassifierModel'].feature_importances_
        feature_names = preprocessor.get_feature_names_out()
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        print(feature_importance_df)
    else:
        print(f"{model_name} does not support feature importances.")
        
train_and_evaluate("Logistic Regression", log_reg)