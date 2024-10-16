# from flask import Flask, request, jsonify
# from joblib import load
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# # Load the pre-trained models
# models = {
#     'model1': load('models/crop_model_rf.joblib'),
#     'model2': load('models/crop_model_dt.joblib'),
#     'model3': load('models/crop_model_svm.joblib'),
#     'model4': load('models/crop_model_lr.joblib'),
#     'model5': load('models/crop_model_knn.joblib')
# }

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     inputs = [
#         data['N'],  # Nitrogen
#         data['P'],  # Phosphorus
#         data['K'],  # Potassium
#         data['temperature'],
#         data['humidity'],
#         data['ph'],
#         data['rainfall']
#     ]
    
#     # Predict with all models
#     predictions = {}
#     for model_name, model in models.items():
#         prediction = model.predict([inputs])
#         predictions[model_name] = prediction[0]

#     return jsonify(predictions)

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, render_template, request, jsonify
# from joblib import load

# app = Flask(__name__)

# # Load your models here
# models = {
#     'model1': load('models/crop_model_rf.joblib'),
#     # 'model2': load('models/crop_model_2.joblib'),
#     # Add other models as necessary
# }

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     features = [[
#         data['nitrogen'], 
#         data['phosphorus'], 
#         data['potassium'], 
#         data['temperature'], 
#         data['humidity'], 
#         data['ph'], 
#         data['rainfall']
#     ]]
    
#     # Assuming you use model1 for predictions, modify as needed
#     prediction = models['model1'].predict(features)[0]
#     return jsonify(prediction=prediction)

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, jsonify, request, render_template
# import os
# import joblib
# import pandas as pd

# app = Flask(__name__)

# # Load model accuracies
# accuracies = {}
# with open(os.path.join(os.path.dirname(__file__), 'model_accuracies.txt'), 'r') as f:
#     for line in f:
#         name, acc = line.strip().split(': ')
#         accuracies[name] = float(acc)

# # Find the best model
# best_model_name = max(accuracies, key=accuracies.get)
# best_model = joblib.load(os.path.join(os.path.dirname(__file__), f"{best_model_name.replace(' ', '_')}.joblib"))

# @app.route('/')
# def index():
#     return render_template('index.html', best_model=best_model_name, accuracy=accuracies[best_model_name])

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     features = [data['N'], data['P'], data['K'], data['temperature'], data['humidity'], data['ph'], data['rainfall']]
#     prediction = best_model.predict([features])
#     return jsonify({'crop': prediction[0]})

# @app.route('/model_comparison')
# def model_comparison():
#     return render_template('model_comparison.html', accuracies=accuracies)

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, jsonify, request, render_template
# import os
# import joblib
# import pandas as pd

# app = Flask(__name__)

# # Load model accuracies
# accuracies = {}
# with open(os.path.join(os.path.dirname(__file__), 'model_accuracies.txt'), 'r') as f:
#     for line in f:
#         name, acc = line.strip().split(': ')
#         accuracies[name] = float(acc)

# # Find the best model
# best_model_name = max(accuracies, key=accuracies.get)
# best_model = joblib.load(os.path.join(os.path.dirname(__file__), f"{best_model_name.replace(' ', '_')}.joblib"))

# @app.route('/')
# def index():
#     return render_template('index.html', best_model=best_model_name, accuracy=accuracies[best_model_name])

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     features = [data['N'], data['P'], data['K'], data['temperature'], data['humidity'], data['ph'], data['rainfall']]
#     prediction = best_model.predict([features])
#     return jsonify({'crop': prediction[0]})

# @app.route('/model_comparison')
# def model_comparison():
#     return render_template('model_comparison.html', accuracies=accuracies)

# if __name__ == '__main__':
#     app.run(debug=True)



# from flask import Flask, jsonify, request, render_template
# import os
# import joblib
# import numpy as np

# app = Flask(__name__)

# # Load model accuracies
# accuracies = {}
# with open(os.path.join(os.path.dirname(__file__), 'model_accuracies.txt'), 'r') as f:
#     for line in f:
#         name, acc = line.strip().split(': ')
#         accuracies[name] = float(acc)

# # Find the best model
# best_model_name = max(accuracies, key=accuracies.get)
# best_model = joblib.load(os.path.join(os.path.dirname(__file__), f"{best_model_name.replace(' ', '_')}.joblib"))
# scaler = joblib.load(os.path.join(os.path.dirname(__file__), 'scaler.joblib'))  # Load the scaler

# @app.route('/')
# def index():
#     return render_template('index.html', best_model=best_model_name, accuracy=accuracies[best_model_name])

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     try:
#         features = np.array([
#             float(data['N']),
#             float(data['P']),
#             float(data['K']),
#             float(data['temperature']),
#             float(data['humidity']),
#             float(data['ph']),
#             float(data['rainfall'])
#         ]).reshape(1, -1)

#         # Scale the features
#         features_scaled = scaler.transform(features)  # Use the loaded scaler

#     except (ValueError, KeyError):
#         return jsonify({'error': 'Invalid input. Please provide valid numeric values for all fields.'}), 400

#     # Predict the crop using the best model
#     prediction = best_model.predict(features_scaled)
    
#     return jsonify({'crop': prediction[0]})

# @app.route('/model_comparison')
# def model_comparison():
#     return render_template('model_comparison.html', accuracies=accuracies)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, jsonify, request, render_template
import os
import joblib
import numpy as np

app = Flask(__name__)

# Load model accuracies
accuracies = {}
with open(os.path.join(os.path.dirname(__file__), 'model_accuracies.txt'), 'r') as f:
    for line in f:
        name, acc = line.strip().split(': ')
        accuracies[name] = float(acc)

# Find the best model
best_model_name = max(accuracies, key=accuracies.get)
best_model_path = os.path.join(os.path.dirname(__file__), f"{best_model_name.replace(' ', '_')}.joblib")

# Check if the model file exists before loading
if os.path.exists(best_model_path):
    best_model = joblib.load(best_model_path)
else:
    raise FileNotFoundError(f"Model file {best_model_path} not found!")

scaler = joblib.load(os.path.join(os.path.dirname(__file__), 'scaler.joblib'))  # Load the scaler

@app.route('/')
def index():
    return render_template('index.html', best_model=best_model_name, accuracy=accuracies[best_model_name])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        features = np.array([
            float(data['N']),
            float(data['P']),
            float(data['K']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]).reshape(1, -1)

        # Scale the features
        features_scaled = scaler.transform(features)  # Use the loaded scaler

    except (ValueError, KeyError):
        return jsonify({'error': 'Invalid input. Please provide valid numeric values for all fields.'}), 400

    # Predict the crop using the best model
    prediction = best_model.predict(features_scaled)
    
    return jsonify({'crop': prediction[0]})

@app.route('/model_comparison')
def model_comparison():
    return render_template('model_comparison.html', accuracies=accuracies)

if __name__ == '__main__':
    app.run(debug=True)





