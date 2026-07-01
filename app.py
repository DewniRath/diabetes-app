import streamlit as st          
import pandas as pd             
import joblib                   


model = joblib.load('diabetes_model.pkl')
feature_names = joblib.load('feature_names.pkl')


st.title('Diabetes Risk Prediction')
st.write('Note: this model was trained on data from women (Pima Indian heritage, age 21+). Predictions for other groups may be unreliable.')
st.write('Enter the patient details below to estimate diabetes risk.')


pregnancies = st.number_input('Pregnancies', min_value=0, value=1)
glucose     = st.number_input('Glucose', min_value=0, value=120)
blood_pressure = st.number_input('Blood Pressure', min_value=0, value=70)
skin_thickness = st.number_input('Skin Thickness', min_value=0, value=20)
insulin     = st.number_input('Insulin', min_value=0, value=80)
bmi         = st.number_input('BMI', min_value=0.0, value=25.0)
pedigree    = st.number_input('Diabetes Pedigree Function', min_value=0.0, value=0.5)
age         = st.number_input('Age', min_value=0, value=30)


if st.button('Predict'):
    
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, pedigree, age]], columns=feature_names)

    
    prediction = model.predict(input_data)[0]

    
    probability = model.predict_proba(input_data)[0][1]

    
    if prediction == 1:
        st.error(f'High risk of diabetes. Estimated probability: {probability:.0%}')
    else:
        st.success(f'Low risk of diabetes. Estimated probability: {probability:.0%}')
