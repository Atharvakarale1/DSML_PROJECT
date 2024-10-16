# # 
# import os
# import pandas as pd
# import joblib
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# from imblearn.over_sampling import SMOTE

# # Load the dataset
# data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'crop_data.csv'))

# # Prepare the features and target variable
# X = data.drop('label', axis=1)  # Features
# y = data['label']  # Target variable

# # Check if the dataset is imbalanced
# class_distribution = y.value_counts()
# print(f"Class distribution:\n{class_distribution}")

# # Use SMOTE to oversample minority classes
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# # Scale the features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Initialize models with hyperparameter tuning
# models = {
#     "Logistic Regression": LogisticRegression(max_iter=300, solver='lbfgs', multi_class='multinomial'),  # Multi-class support
#     "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
#     "Support Vector Classifier": SVC(kernel='rbf', gamma='auto', probability=True, random_state=42),
#     "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
#     "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
# }

# # Train the models and store accuracies using cross-validation
# accuracies = {}
# for model_name, model in models.items():
#     # Apply cross-validation for more robust performance estimates
#     cv_scores = cross_val_score(model, X_train, y_train, cv=5)
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)
#     accuracy = accuracy_score(y_test, predictions) * 100
#     accuracies[model_name] = accuracy
#     print(f"{model_name} Accuracy: {accuracy:.2f}% (CV Average: {cv_scores.mean()*100:.2f}%)")

# # Save model accuracies to a file
# with open(os.path.join(os.path.dirname(__file__), 'model_accuracies.txt'), 'w') as f:
#     for name, acc in accuracies.items():
#         f.write(f"{name}: {acc:.4f}\n")

# # Save the best model based on accuracy
# best_model_name = max(accuracies, key=accuracies.get)
# best_model = models[best_model_name]
# joblib.dump(best_model, os.path.join(os.path.dirname(__file__), f"{best_model_name.replace(' ', '_')}.joblib"))

# # Print success message
# print("All models trained successfully!")
# print(f"Best model: {best_model_name} with accuracy: {accuracies[best_model_name]:.2f}%")


# import os
# import pandas as pd
# import joblib
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from imblearn.over_sampling import SMOTE
# import numpy as np

# # Load the dataset
# data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'crop_data.csv'))

# # Prepare the features and target variable
# X = data.drop('label', axis=1)  # Features
# y = data['label']  # Target variable

# # Handle imbalanced data using SMOTE
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# # Scale the features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Initialize models with some fixed parameters for faster training
# models = {
#     "Logistic Regression": LogisticRegression(max_iter=200),
#     "Random Forest": RandomForestClassifier(),
#     "Gradient Boosting": GradientBoostingClassifier(),
#     "Support Vector Classifier": SVC(C=1, kernel='linear'),  # Fixed hyperparameters to save time
#     "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3)  # Fixed hyperparameters to save time
# }

# # Reduced hyperparameter grid for RandomForest and GradientBoosting only
# param_grids = {
#     "Random Forest": {
#         'n_estimators': [50, 100],
#         'max_depth': [None, 10]
#     },
#     "Gradient Boosting": {
#         'n_estimators': [50, 100],
#         'learning_rate': [0.1, 0.01],
#         'max_depth': [3]
#     }
# }

# # Train the models and store accuracies
# accuracies = {}
# for model_name, model in models.items():
#     if model_name in param_grids:
#         # Apply RandomizedSearchCV only for RandomForest and GradientBoosting
#         random_search = RandomizedSearchCV(model, param_grids[model_name], n_iter=3, cv=3, scoring='accuracy', random_state=42)
#         random_search.fit(X_train_scaled, y_train)
#         best_model = random_search.best_estimator_
#         print(f"{model_name} Best Parameters: {random_search.best_params_}")
#     else:
#         # For simpler models, no hyperparameter tuning
#         best_model = model.fit(X_train_scaled, y_train)

#     predictions = best_model.predict(X_test_scaled)
#     accuracy = accuracy_score(y_test, predictions) * 100
#     accuracies[model_name] = accuracy

#     print(f"{model_name} Accuracy: {accuracy:.2f}%")
#     print(f"Classification Report for {model_name}:\n{classification_report(y_test, predictions)}")

# # Save model accuracies to a file
# with open(os.path.join(os.path.dirname(__file__), 'model_accuracies.txt'), 'w') as f:
#     for name, acc in accuracies.items():
#         f.write(f"{name}: {acc:.4f}\n")

# # Save the best model based on accuracy
# best_model_name = max(accuracies, key=accuracies.get)
# best_model = models[best_model_name]
# joblib.dump(best_model, os.path.join(os.path.dirname(__file__), f"{best_model_name.replace(' ', '_')}.joblib"))

# # Save the scaler
# joblib.dump(scaler, os.path.join(os.path.dirname(__file__), 'scaler.joblib'))

# # Print success message
# print("All models trained successfully!")
# print(f"Best model: {best_model_name} with accuracy: {accuracies[best_model_name]:.2f}%")


import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import numpy as np

# Load the dataset
data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'crop_data.csv'))

# Prepare the features and target variable
X = data.drop('label', axis=1)  # Features
y = data['label']  # Target variable

# Handle imbalanced data using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models with some fixed parameters for faster training
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Support Vector Classifier": SVC(C=1, kernel='linear'),  # Fixed hyperparameters to save time
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3)  # Fixed hyperparameters to save time
}

# Reduced hyperparameter grid for RandomForest and GradientBoosting only
param_grids = {
    "Random Forest": {
        'n_estimators': [50, 100],
        'max_depth': [None, 10]
    },
    "Gradient Boosting": {
        'n_estimators': [50, 100],
        'learning_rate': [0.1, 0.01],
        'max_depth': [3]
    }
}

# Train the models and store accuracies
accuracies = {}
for model_name, model in models.items():
    if model_name in param_grids:
        # Apply RandomizedSearchCV only for RandomForest and GradientBoosting
        random_search = RandomizedSearchCV(model, param_grids[model_name], n_iter=3, cv=3, scoring='accuracy', random_state=42)
        random_search.fit(X_train_scaled, y_train)
        best_model = random_search.best_estimator_
        print(f"{model_name} Best Parameters: {random_search.best_params_}")
    else:
        # For simpler models, no hyperparameter tuning
        best_model = model.fit(X_train_scaled, y_train)

    predictions = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions) * 100
    accuracies[model_name] = accuracy

    print(f"{model_name} Accuracy: {accuracy:.2f}%")
    print(f"Classification Report for {model_name}:\n{classification_report(y_test, predictions)}")

    # Save the model for each trained model
    joblib.dump(best_model, os.path.join(os.path.dirname(__file__), f"{model_name.replace(' ', '_')}.joblib"))

# Save model accuracies to a file
with open(os.path.join(os.path.dirname(__file__), 'model_accuracies.txt'), 'w') as f:
    for name, acc in accuracies.items():
        f.write(f"{name}: {acc:.4f}\n")

# Save the scaler
joblib.dump(scaler, os.path.join(os.path.dirname(__file__), 'scaler.joblib'))

# Print success message
print("All models trained successfully!")
print(f"Best model: {max(accuracies, key=accuracies.get)} with accuracy: {max(accuracies.values()):.2f}%")





