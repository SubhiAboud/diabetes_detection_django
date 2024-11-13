
import gradio as gr
import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load('models and data/Pickle/lr.sav')
scaler= joblib.load('models and data/Pickle/scaler.sav')
def predict_diabetes(glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age):
    # Create a numpy array from the input values
    input_data = np.array([[glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    
    # Make a prediction using the model
    scaled_data=scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    #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
    # Return the prediction result
    if prediction[0] == 1:
        return "The model predicts that the patient has diabetes."
    else:
        return "The model predicts that the patient does not have diabetes."

# Create the Gradio interface
interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Glucose"),
        gr.Number(label="Blood Pressure"),
        gr.Number(label="Skin Thickness"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="Diabetes Pedigree Function"),
        gr.Number(label="Age")
    ],
    outputs="text",
    title="Diabetes Detection App",
    description="Enter the patient's medical information to predict if they have diabetes."
)

# Launch the Gradio app
interface.launch()
