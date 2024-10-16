# Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import warnings
import os
warnings.filterwarnings("ignore")

# Load dataset
data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'crop_data.csv'))

# Exploratory Data Analysis (EDA)
def perform_eda(data):
    print("First 5 Rows of Dataset:\n", data.head())
    print("\nBasic Information of the Dataset:\n", data.info())
    print("\nMissing Values:\n", data.isnull().sum())
    print("\nStatistical Overview:\n", data.describe())
    
    # Correlation Heatmap (for numerical columns only)
    numerical_data = data.select_dtypes(include=[np.number])  # Only select numerical columns
    plt.figure(figsize=(10, 6))
    sns.heatmap(numerical_data.corr(), annot=True, cmap="coolwarm")
    plt.title('Correlation Heatmap of Numerical Features')  # Title for heatmap
    plt.show()

    # Sample a smaller subset for visualization to reduce time
    sample_data = data.sample(frac=0.1, random_state=42)  # Use 10% of the data for heavy plots

    # Pair Plot (on the sampled data)
    sns.pairplot(sample_data, hue='label')  # Replace 'label' with your target column
    plt.title('Pair Plot of Sampled Data (10% of dataset)')  # Title for pair plot
    plt.show()

    # Feature Distribution Plots
    sample_data.hist(bins=15, figsize=(15, 10), layout=(4, 3))
    plt.suptitle('Feature Distributions')  # Title for feature distribution
    plt.show()

    return data

# Perform EDA and prepare the data
df = perform_eda(data)

# Select Features and Target
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]  # Replace with your feature columns
y = df['label']  # Replace with your target column

# Split Data into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define Models
models = {
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier()
}

# Model Training and Evaluation
def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f'{name} Accuracy: {accuracy:.4f}')
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')  # Title for confusion matrix
        plt.show()

        # Classification Report
        print(f'\n{name} Classification Report:\n', classification_report(y_test, y_pred))

        # ROC/AUC Curve
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=model.classes_[1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} AUC = {roc_auc:.2f}')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for All Models')  # Title for ROC curve
    plt.legend(loc="lower right")
    plt.show()

    return results

# Evaluate models without tuning
results = evaluate_models(models, X_train, X_test, y_train, y_test)

# Model Fine-Tuning with GridSearchCV and RandomizedSearchCV
def fine_tune_model(model, param_grid, search_type='grid'):
    if search_type == 'grid':
        search = GridSearchCV(model, param_grid, cv=5, verbose=2, n_jobs=-1)
    else:
        search = RandomizedSearchCV(model, param_grid, n_iter=100, cv=5, verbose=2, n_jobs=-1, random_state=42)
    
    search.fit(X_train, y_train)
    print(f"Best Parameters: {search.best_params_}")
    best_model = search.best_estimator_
    
    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    return best_model

# Example: Fine-tuning Random Forest with GridSearchCV
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

print("Fine-tuning Random Forest...")
best_rf_model = fine_tune_model(RandomForestClassifier(), param_grid_rf, search_type='grid')

# Example: Fine-tuning SVM with RandomizedSearchCV
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': [0.001, 0.01, 0.1]
}

print("Fine-tuning SVM...")
best_svm_model = fine_tune_model(SVC(probability=True), param_grid_svm, search_type='random')
