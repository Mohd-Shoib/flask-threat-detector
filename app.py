from flask import Flask, render_template, request
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the same architecture used during training
autoencoder = Sequential([
    Dense(16, activation='relu', input_shape=(6,)),
    Dense(8, activation='relu'),
    Dense(16, activation='relu'),
    Dense(6, activation='linear'),
])
autoencoder.compile(optimizer='adam', loss='mse')

# Load pre-trained weights
autoencoder.load_weights("model/autoencoder_weights.weights.h5")



app = Flask(__name__)

# Load model, scaler, threshold
autoencoder.load_weights("model/autoencoder_weights.weights.h5")
scaler = pickle.load(open('model/scaler.pkl', 'rb'))
threshold = pickle.load(open('model/threshold.pkl', 'rb'))



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Input fields
        fields = ['hour', 'dayofweek', 'attachments', 'file_access_count',
                  'unique_files_accessed', 'device_count']
        input_data = []

        for field in fields:
            val = request.form.get(field)
            if val is None or val.strip() == '':
                raise ValueError(f"Missing input for: {field}")
            input_data.append(float(val))

        input_array = np.array([input_data])
        input_scaled = scaler.transform(input_array)

        # Reconstruct & calculate MSE
        reconstruction = autoencoder.predict(input_scaled)
        mse = np.mean(np.square(input_scaled - reconstruction))

        # Label prediction
        label = "Potential Threat" if mse > threshold else "Normal User"

        # Rule-based safety check
        if(8 <= input_data[0] <= 22 and
            0 <= input_data[1] <= 4 and
            input_data[2] <= 3):
                label = "Normal User"
        if (input_data[3] > 300 and 
            input_data[4] > 100 and
            input_data[5] > 5):
            label = "Potential Threat"

        #Shap plot
        shap_img_path = ""
        if label == "Potential Threat":
            shap_img_path = "static/shap_images/shap_summary.png"


        #Explaination Needed
        explanation = ""
        if label == "Potential Threat":
            explanation = "Anomaly likely due to: "
            if input_data[0] < 8 or input_data[0] > 22:
                explanation += "Odd Hour; "
            if input_data[1] == 5 or input_data[1] == 6 :
                explanation += "Access on Holidays; "
            if input_data[2] > 3:
                explanation += "Too Many Attachments; "
            if input_data[3] > 300:
                explanation += "High File Access;"
            if input_data[4] > 100:
                explanation += "Too Many Unique Files; "
            if input_data[5] > 5:
                explanation += "Excessive Device Count; "
        else :
            explanation = "It is a Good User"


        return render_template(
            'index.html',
            prediction=label,
            explanation=explanation,
            shap_image=shap_img_path,
            input_data=dict(zip(fields, input_data)))


    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
