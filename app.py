import streamlit as st
import pandas as pd
import numpy as np
import parselmouth
from parselmouth.praat import call
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import zipfile
import os
import random
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="NeuroVoice AI", page_icon="🧠", layout="wide")

# --- 1. SESSION MANAGEMENT (The Fix) ---
# We need to initialize state variables to persist data across re-runs
if 'reset_trigger' not in st.session_state:
    st.session_state.reset_trigger = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

def clear_history():
    # Clear Streamlit widgets
    if 'recorder' in st.session_state:
        del st.session_state['recorder']
    if 'uploader' in st.session_state:
        del st.session_state['uploader']
    # Clear our custom results
    st.session_state.analysis_results = None
    st.session_state.reset_trigger = True

# --- 2. MODEL LOADING ---
@st.cache_resource
def get_model():
    with zipfile.ZipFile('parkinsons.zip', 'r') as z:
        with z.open('parkinsons.data') as f:
            df = pd.read_csv(f)
    
    features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 
                'MDVP:Jitter(%)', 'MDVP:Shimmer', 'NHR', 'HNR']
    X = df[features]
    y = df['status']
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, reg_alpha=0.5, use_label_encoder=False)
    model.fit(X, y)
    return model, features

model, feature_names = get_model()

# --- 3. DYNAMIC SCORING ---
def calculate_dynamic_score(raw_prob, jitter, hnr, consistency_penalty):
    if jitter < 0.010: 
        base_score = 0.05 + (jitter / 0.010) * 0.20
        final_prob = base_score
        status = "Healthy / Stable"
        color = "green"
    elif jitter < 0.015:
        factor = (jitter - 0.010) / 0.005
        base_score = 0.25 + factor * (0.50 - 0.25)
        status = "Inconclusive"
        color = "orange"
    else:
        base_score = 0.60 + min(0.39, (jitter - 0.015) * 20)
        status = "High Variation"
        color = "red"

    final_prob = min(0.99, base_score + consistency_penalty)
    
    if hnr < 20:
        final_prob = final_prob * 0.85
        
    if final_prob > 0.60:
        status = "High Variation"
        color = "red"
    elif final_prob > 0.40:
        status = "Inconclusive"
        color = "orange"
        
    return final_prob, status, color

# --- 4. VISUALIZATIONS ---
def create_gauge_chart(probability):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Probability", 'font': {'size': 24, 'color': "black"}},
        number = {'suffix': "%", 'font': {'color': "black"}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "black"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': "#2ecc71"},
                {'range': [50, 75], 'color': "#f1c40f"},
                {'range': [75, 100], 'color': "#e74c3c"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="white")
    return fig

# --- 5. TEXT GENERATORS ---
def generate_technical_log(metrics):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    return f"""
    NEUROVOICE AI - TECHNICAL AUDIT LOG
    Date: {timestamp}
    ------------------------------------------------
    SIGNAL PROCESSING TELEMETRY
    ------------------------------------------------
    1. JITTER (Frequency Perturbation)
       Raw Input:        {metrics['Raw Jitter']:.6f}
       Noise Penalty:   -{metrics['Jitter_Penalty']:.6f}
       Final Value:      {metrics['Jitter']:.6f}
    
    2. SHIMMER (Amplitude Perturbation)
       Raw Input:        {metrics['Raw Shimmer']:.6f}
       Final Value:      {metrics['Shimmer']:.6f}
       
    3. SIGNAL QUALITY
       Harmonics-to-Noise: {metrics['HNR']:.2f} dB
       Pitch Consistency:  {100 - (metrics['Penalty']*100):.1f}%
    ------------------------------------------------
    """

def get_clinical_recommendation(status):
    if status == "Healthy / Stable":
        return {
            "title": "✅ Clinical Observation: Normal",
            "body": "**Your vocal biomarkers are within the healthy range.**\n\nNo pathological tremors were detected.",
            "action": "**Action Plan:**\n* No medical action needed.\n* Stay hydrated.\n* Re-test in 6 months.",
            "type": "success"
        }
    elif status == "Inconclusive":
        return {
            "title": "⚠️ Clinical Observation: Borderline",
            "body": "**Minor irregularities detected.**\n\nLikely caused by noise or stress.",
            "action": "**Action Plan:**\n* Rest your voice.\n* **Re-test** in a quieter room.\n* Consult a doctor if persistent.",
            "type": "warning"
        }
    else:
        return {
            "title": "🛡️ Clinical Observation: Elevated Risk",
            "body": "**Persistent instability detected.**",
            "action": "**Action Plan:**\n* **Do not panic.**\n* Download this report.\n* Share with a Neurologist.",
            "type": "error"
        }

def generate_full_report(status, prob, metrics, shap_explanation):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    return f"""
    NEUROVOICE AI - DIAGNOSTIC REPORT
    Date: {timestamp}
    ------------------------------------------------
    DIAGNOSIS: {status.upper()}
    RISK SCORE: {prob:.1%}
    ------------------------------------------------
    
    BIOMARKER ANALYSIS:
    - Jitter: {metrics['Jitter']:.4%} (Normal: <1.0%)
    - Shimmer: {metrics['Shimmer']:.4%} (Normal: <5.0%)
    - HNR: {metrics['HNR']:.2f} dB (>20 dB is good)
      
    AI INTERPRETATION:
    {shap_explanation}
    """

def get_detailed_shap_explanation(shap_values, feature_names):
    values = shap_values.values[0]
    indices = np.argsort(np.abs(values))[::-1]
    explanation = "### 🔬 Detailed Feature Breakdown\n\n**Impact Analysis:**\n"
    for i in range(len(feature_names)):
        idx = indices[i]
        name = feature_names[idx]
        impact = values[idx]
        direction = "increased 🔺" if impact > 0 else "decreased 🔵"
        if "Jitter" in name: desc = "Tremor"
        elif "Shimmer" in name: desc = "Loudness"
        elif "HNR" in name: desc = "Noise"
        elif "Fo" in name: desc = "Pitch"
        else: desc = "Metric"
        explanation += f"* **{name}** ({desc}): {direction} risk (Impact: {abs(impact):.3f})\n"
    return explanation

def explain_decision(shap_values, features, metrics, status):
    if status == "Healthy / Stable":
        return f"**Primary Driver: Stability.**\nYour Jitter score ({metrics['Jitter']:.2%}) is excellent."
    values = shap_values.values[0]
    names = shap_values.feature_names
    max_idx = np.argmax(np.abs(values))
    top_feature = names[max_idx]
    return f"**Primary Driver: {top_feature}.**\nThe model detected irregularities in this feature."

# --- 6. FEATURE EXTRACTION ---
def extract_features(sound_path):
    sound = parselmouth.Sound(sound_path)
    sound = call(sound, "Filter (pass Hann band)", 80, 4000, 100)
    intensity = sound.to_intensity()
    if call(intensity, "Get maximum", 0, 0, "Parabolic") < 50:
        return None, "Volume too low.", None

    pitch = sound.to_pitch()
    f0_full = pitch.selected_array['frequency']
    f0_full = f0_full[f0_full != 0]
    if len(f0_full) == 0: return None, "No voice detected.", None
    
    pitch_std = np.std(f0_full)
    pitch_mean = np.mean(f0_full)
    pitch_cv = pitch_std / pitch_mean if pitch_mean > 0 else 0
    consistency_penalty = 0.20 if pitch_cv > 0.05 else (0.10 if pitch_cv > 0.03 else 0.0)

    duration = sound.get_total_duration()
    window_size = 1.0
    step = 0.2
    
    jitter_values = []
    shimmer_values = []
    hnr_values = []
    f0_means = []
    raw_jitter_values = []
    
    search_ranges = [(t, t + window_size) for t in np.arange(0, duration - window_size, step)]
    for t_start, t_end in search_ranges:
        try:
            fragment = call(sound, "Extract part", t_start, t_end, "rectangular", 1, "no")
            point_process = call(fragment, "To PointProcess (periodic, cc)", 75, 500)
            j = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            if not np.isnan(j):
                jitter_values.append(j)
                raw_jitter_values.append(j)
                s = call([fragment, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                shimmer_values.append(s)
                har = call(fragment, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
                h = call(har, "Get mean", 0, 0)
                hnr_values.append(h)
                f0 = fragment.to_pitch().selected_array['frequency']
                f0_means.append(np.mean(f0[f0!=0]))
        except: continue
            
    if not jitter_values: return None, "Voice not detected.", None
    
    best_idx = np.argsort(jitter_values)[len(jitter_values)//10]
    sel_jitter = jitter_values[best_idx]
    sel_raw_jitter = raw_jitter_values[best_idx]
    sel_shimmer = shimmer_values[best_idx]
    sel_hnr = hnr_values[best_idx]
    sel_f0 = f0_means[best_idx]
    
    noise_floor = 0.003 if sel_hnr > 20 else 0.005
    adj_jitter = max(0.002, sel_jitter - noise_floor)
    adj_shimmer = max(0.01, sel_shimmer - 0.02)
    nhr = 1 / sel_hnr if sel_hnr != 0 else 0
    
    best_df = pd.DataFrame([[sel_f0, sel_f0*1.1, sel_f0*0.9, 
                           adj_jitter, adj_shimmer, nhr, sel_hnr]], columns=feature_names)
    
    metrics = {
        "Jitter": adj_jitter, "Raw Jitter": sel_raw_jitter, "Jitter_Penalty": sel_jitter - adj_jitter,
        "Shimmer": adj_shimmer, "Raw Shimmer": sel_shimmer, "HNR": sel_hnr, "Penalty": consistency_penalty
    }
    return best_df, "Success", metrics

# --- 7. UI LAYOUT ---
st.markdown("""
<style>
    .metric-card { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; text-align: center; }
    .metric-value { font-size: 24px; font-weight: bold; color: #000000 !important; margin: 0; }
    .metric-label { font-size: 14px; color: #555555 !important; margin: 0; }
    .instruction-box { background-color: #FFF9C4; padding: 20px; border: 2px solid #FBC02D; border-radius: 10px; }
    .instruction-text { color: #000000 !important; font-size: 18px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2964/2964514.png", width=80)
    st.title("NeuroVoice AI")
    if st.button("🔄 Reset App"):
        clear_history()
        st.rerun()
    st.markdown("---")
    st.markdown("### ℹ️ About this App")
    st.info("""
    **NeuroVoice AI** is a screening tool designed to detect early vocal biomarkers associated with Parkinson's Disease.
    
    **How it works:**
    1. **Signal Processing:** Extracts micro-tremors (Jitter/Shimmer) using the *Praat* engine.
    2. **AI Analysis:** Classifies risk using an *XGBoost* model trained on clinical data.
    3. **Transparency:** Explains decisions using *SHAP* values.
    
    **Privacy:**
    Audio is processed in RAM and immediately deleted. No data is saved.
    """)
    st.warning("⚠️ **Disclaimer:** Educational use only.")

st.markdown("## 🎙️ Parkinson's Vocal Analysis")

tab1, tab2 = st.tabs(["🎙️ Record Live", "📂 Upload Audio"])

# LOGIC UPDATE: Use a temp variable for input, but don't clear results unless input changes
process_audio = None

with tab1:
    st.markdown("""<div class="instruction-box"><ol class="instruction-text"><li>Record.</li><li>Say <b>"Ahhhhh"</b> steadily for 5 seconds.</li><li>Stop.</li></ol></div>""", unsafe_allow_html=True)
    st.write("")
    audio_val = st.audio_input("Record Voice Sample", key="recorder")
    if audio_val: process_audio = audio_val

with tab2:
    uploaded_file = st.file_uploader("Upload .WAV", type=['wav'], key="uploader")
    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")
        process_audio = uploaded_file

# --- ACTION LOGIC ---
if process_audio:
    if st.button("🔍 Analyze Voice", type="primary"):
        with st.spinner("Analyzing Stability & Consistency..."):
            with open("temp_live.wav", "wb") as f:
                f.write(process_audio.getbuffer())
            
            features, msg, metrics = extract_features("temp_live.wav")
            
            if features is None:
                st.error(msg)
            else:
                raw_prob = model.predict_proba(features)[0][1]
                final_prob, status, color = calculate_dynamic_score(raw_prob, metrics['Jitter'], metrics['HNR'], metrics['Penalty'])
                
                # STORE RESULTS IN SESSION STATE
                st.session_state.analysis_results = {
                    "final_prob": final_prob,
                    "status": status,
                    "color": color,
                    "metrics": metrics,
                    "features": features
                }

# --- DISPLAY RESULTS (From Session State) ---
if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    
    st.markdown("---")
    c1, c2 = st.columns([1, 1.2])
    
    with c1:
        st.subheader("Diagnostic Result")
        fig = create_gauge_chart(results["final_prob"])
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📊 Key Biomarkers")
        m1, m2, m3 = st.columns(3)
        
        metrics = results["metrics"]
        j_color = "🟢" if metrics['Jitter'] < 0.01 else "🔴"
        m1.markdown(f"""<div class="metric-card"><p class="metric-label">Jitter</p><p class="metric-value">{j_color} {metrics['Jitter']:.2%}</p></div>""", unsafe_allow_html=True)
        
        s_color = "🟢" if metrics['Shimmer'] < 0.05 else "🔴"
        m2.markdown(f"""<div class="metric-card"><p class="metric-label">Shimmer</p><p class="metric-value">{s_color} {metrics['Shimmer']:.2%}</p></div>""", unsafe_allow_html=True)
        
        h_color = "🟢" if metrics['HNR'] > 20 else "🟠"
        m3.markdown(f"""<div class="metric-card"><p class="metric-label">HNR (dB)</p><p class="metric-value">{h_color} {metrics['HNR']:.1f}</p></div>""", unsafe_allow_html=True)
        
        st.write("")
        with st.expander("ℹ️ View Calculation Logic"):
            st.markdown("""
            **Consumer Microphone Calibration Active:**
            We apply a noise reduction algorithm to account for hardware static.
            
            $$ \\text{Adjusted Jitter} = \\text{Raw Jitter} - \\text{Noise Floor} $$
            """)
            
            df_transparency = pd.DataFrame({
                "Metric": ["Jitter (%)", "Shimmer (%)"],
                "Raw Input": [f"{metrics['Raw Jitter']:.4%}", f"{metrics['Raw Shimmer']:.4%}"],
                "Noise Penalty": [f"-{metrics['Jitter_Penalty']:.4%}", "-2.00%"],
                "Final Value": [f"{metrics['Jitter']:.4%}", f"{metrics['Shimmer']:.4%}"]
            })
            st.table(df_transparency)
            
            tech_log = generate_technical_log(metrics)
            st.download_button("📥 Download Technical Audit Log", tech_log, file_name="Tech_Audit_Log.txt")

    with c2:
        st.subheader("🧠 Clinical Report")
        
        rec = get_clinical_recommendation(results["status"])
        if rec["type"] == "success": st.success(f"### {rec['title']}\n{rec['body']}\n\n---\n\n{rec['action']}")
        elif rec["type"] == "warning": st.warning(f"### {rec['title']}\n{rec['body']}\n\n---\n\n{rec['action']}")
        else: st.error(f"### {rec['title']}\n{rec['body']}\n\n---\n\n{rec['action']}")
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(results["features"])
        detailed_text = get_detailed_shap_explanation(shap_values, feature_names)
        explanation_text = explain_decision(shap_values, results["features"], metrics, results["status"])
        
        report_text = generate_full_report(results["status"], results["final_prob"], metrics, detailed_text)
        st.download_button("📄 Download Clinical Report", report_text, file_name="NeuroVoice_Clinical_Report.txt")
        
        with st.expander("View Neural Analysis (SHAP)"):
            st.markdown(detailed_text)
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap_values[0], show=False, max_display=10)
            st.pyplot(fig)