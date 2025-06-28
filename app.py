from flask import Flask, render_template, request
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler



app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('model/cert_model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))



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

        input_scaled = scaler.transform([input_data])
        prediction = model.predict(input_scaled)
        label = "Potential Threat" if prediction[0] == -1 else "Normal User"


        # Rule-based safety check
        if(8 <= input_data[0] <= 22 and
            0 <= input_data[1] <= 4 and
            input_data[2] <= 3):
                label = "Normal User"
        else: label = "Potential Threat"

        if (input_data[3] > 300 or 
            input_data[4] > 100 or
            input_data[5] > 5):
            label = "Potential Threat"
        else: label = "Normal User"

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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

