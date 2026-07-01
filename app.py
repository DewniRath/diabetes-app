import streamlit as st
import pandas as pd
import joblib

# ---------- Page config (browser tab title + centred layout) ----------
st.set_page_config(
    page_title="Diabetes Risk Screening",
    page_icon="🩺",
    layout="centered",
)

# ---------- Load the trained model and the feature order ----------
model = joblib.load('diabetes_model.pkl')
feature_names = joblib.load('feature_names.pkl')

# ---------- Custom styling (CSS injected into the page) ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

/* apply the font across the app */
html, body, [class*="css"], .stApp, .stMarkdown, .stNumberInput label {
    font-family: 'Manrope', sans-serif;
}

/* soft mint page background */
.stApp { background: #F4F9F8; }

/* gradient hero banner */
.hero {
    background: linear-gradient(135deg, #0F766E 0%, #0891B2 100%);
    padding: 2.2rem 2rem;
    border-radius: 20px;
    color: #ffffff;
    box-shadow: 0 10px 30px rgba(15,118,110,0.25);
    margin-bottom: 1.4rem;
}
.hero h1 { font-size: 1.9rem; font-weight: 800; margin: 0 0 0.35rem 0; color: #fff; line-height: 1.15; }
.hero p  { font-size: 1rem; margin: 0; color: #E0F2F1; font-weight: 500; }
.chip {
    display: inline-block; margin-top: 1rem;
    background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.30);
    padding: 0.35rem 0.85rem; border-radius: 999px;
    font-size: 0.8rem; font-weight: 600; color: #fff;
}

/* section heading + note */
.section-title { font-size: 1.1rem; font-weight: 700; color: #0F2620; margin: 0.4rem 0 0.1rem 0; }
.section-sub   { font-size: 0.88rem; color: #5B7772; margin-bottom: 0.7rem; }

/* action button */
.stButton > button {
    background: linear-gradient(135deg, #0F766E 0%, #0891B2 100%);
    color: #ffffff; font-weight: 700; font-size: 1.05rem;
    border: none; border-radius: 12px; padding: 0.7rem 1rem; width: 100%;
    transition: transform 0.08s ease, box-shadow 0.2s ease;
    box-shadow: 0 6px 18px rgba(15,118,110,0.28);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(15,118,110,0.35); color: #ffffff;
}

/* colour-coded result card */
.result-card { border-radius: 16px; padding: 1.6rem; margin-top: 1.2rem; text-align: center; }
.result-card .label  { font-size: 1.2rem; font-weight: 700; margin: 0; }
.result-card .prob   { font-size: 3rem; font-weight: 800; line-height: 1; margin: 0.25rem 0; }
.result-card .advice { font-size: 0.95rem; margin-top: 0.5rem; font-weight: 500; }
.result-low  { background: #ECFDF5; border: 1px solid #A7F3D0; color: #065F46; }
.result-mod  { background: #FFFBEB; border: 1px solid #FDE68A; color: #92400E; }
.result-high { background: #FEF2F2; border: 1px solid #FECACA; color: #991B1B; }

/* footer */
.footer { margin-top: 2rem; text-align: center; font-size: 0.8rem; color: #8A9E9A; }
</style>
""", unsafe_allow_html=True)

# ---------- Hero banner ----------
st.markdown("""
<div class="hero">
    <h1>🩺 Diabetes Risk Screening</h1>
    <p>Enter a few routine health measurements to estimate diabetes risk.</p>
    <span class="chip">Screening aid • not a medical diagnosis</span>
</div>
""", unsafe_allow_html=True)

# ---------- Scope note (honest disclosure = responsible-use marks) ----------
st.markdown(
    '<div class="section-sub">Note: this model was trained on data from women '
    '(Pima Indian heritage, age 21+). Predictions for other groups may be unreliable.</div>',
    unsafe_allow_html=True
)

# ---------- Input fields, arranged in two columns ----------
st.markdown('<div class="section-title">Patient measurements</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    pregnancies    = st.number_input('Pregnancies', min_value=0, value=1)
    glucose        = st.number_input('Glucose', min_value=0, value=120)
    blood_pressure = st.number_input('Blood Pressure', min_value=0, value=70)
    skin_thickness = st.number_input('Skin Thickness', min_value=0, value=20)
with col2:
    insulin  = st.number_input('Insulin', min_value=0, value=80)
    bmi      = st.number_input('BMI', min_value=0.0, value=25.0)
    pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, value=0.5)
    age      = st.number_input('Age', min_value=0, value=30)

# ---------- Prediction ----------
if st.button('Check risk'):
    # pack inputs into one row, matching the training column order
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, pedigree, age]], columns=feature_names)

    # probability of diabetes (0 to 1)
    probability = model.predict_proba(input_data)[0][1]
    pct = f"{probability:.0%}"

    # show a colour-coded result card based on three risk bands
    if probability < 0.3:
        st.markdown(f"""
        <div class="result-card result-low">
            <p class="label">Low risk</p>
            <p class="prob">{pct}</p>
            <p class="advice">Estimated probability of diabetes. Keep up healthy habits.</p>
        </div>""", unsafe_allow_html=True)
    elif probability < 0.6:
        st.markdown(f"""
        <div class="result-card result-mod">
            <p class="label">Moderate risk</p>
            <p class="prob">{pct}</p>
            <p class="advice">Estimated probability of diabetes. Consider speaking to a doctor.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card result-high">
            <p class="label">High risk</p>
            <p class="prob">{pct}</p>
            <p class="advice">Estimated probability of diabetes. Please consult a healthcare professional.</p>
        </div>""", unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown(
    '<div class="footer">Academic screening demonstration • Random Forest model • '
    'Not a substitute for professional medical advice.</div>',
    unsafe_allow_html=True
)
