"""Streamlit UI.  Run:  streamlit run app.py"""
import streamlit as st
from PIL import Image

import predictor
from report import DISCLAIMER, build_report
import cv2
import numpy as np

st.set_page_config(page_title="Medical Diagnosis Assistant", page_icon="🩺")
st.title("🩺 Medical Diagnosis Assistant")
st.warning(DISCLAIMER)

name = st.text_input("Patient name (optional)", "Anonymous")
tab_sym, tab_img, tab_skin, tab_chat, tab_doc = st.tabs(
    ["Symptoms", "Chest X-ray", "Skin", "Chatbot", "Find a Doctor"])

symptoms, sym_preds, xray = [], None, None

with tab_sym:
    try:
        options = predictor.symptom_list()
        symptoms = st.multiselect("Select symptoms", options)
        if symptoms:
            sym_preds = predictor.predict_symptoms(symptoms)
            for disease, p in sym_preds:
                st.write(f"**{disease}**")
                st.progress(p, text=f"{p:.0%}")
    except FileNotFoundError:
        st.info("Symptom model not found. Run `python train_symptoms.py` first.")

with tab_img:
    up = st.file_uploader("Upload chest X-ray", type=["jpg", "jpeg", "png"])
    if up:
        img = Image.open(up)
        st.image(img, caption="Uploaded image", width=320)
        if st.button("Analyze X-ray"):
            try:
                xray = predictor.predict_xray(img)
                st.session_state["xray"] = xray
            except (FileNotFoundError, OSError):
                st.info("X-ray model not found. Run `python train.py` first.")
    xray = st.session_state.get("xray")
    if xray:
        st.subheader(f"{xray['label']}  ({xray['confidence']:.0%})")
        for k, v in xray["probabilities"].items():
            st.progress(v, text=f"{k}: {v:.0%}")

if symptoms or xray:
    st.download_button("Download PDF report",
                       build_report(name, symptoms, sym_preds, xray),
                       file_name="diagnosis_report.pdf", mime="application/pdf")
with tab_skin:
    st.subheader("Skin lesion analysis")
    skin_file = st.file_uploader("Upload a skin lesion photo",
                                 type=["jpg", "jpeg", "png"], key="skin_upload")
    if skin_file is not None:
        img = cv2.imdecode(np.frombuffer(skin_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=300)
        if st.button("Analyze skin lesion", key="skin_btn"):
            try:
                from skin_predictor import predict_skin
                result = predict_skin(img)
                st.session_state["last_skin"] = result["Label"]
                st.write(f"**Result: {result['Label']}**")
                for label in ("Malignant", "Benign"):
                    st.write(f"{label}: {result[label]:.0%}")
                    st.progress(float(result[label]))
                st.info("Educational model only, not a diagnosis. "
                        "Any suspicious lesion should be checked by a dermatologist.")
            except Exception as e:
                st.error(f"Skin model error: {e}")
with tab_chat:
    st.subheader("Health chatbot")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for role, text in st.session_state.chat_history:
        st.chat_message(role).markdown(text)
    question = st.chat_input("Ask a health question...", key="chat_box")
    if question:
        from chatbot import chat
        st.session_state.chat_history.append(("user", question))
        st.session_state.chat_history.append(("assistant", chat(question)))
        st.rerun()
with tab_doc:
    st.subheader("Doctor recommendation")
    st.caption("Demo only: the doctors listed are sample data, not real people.")
    cond = st.text_input("Condition or predicted result",
                         value=st.session_state.get("last_skin", ""), key="doc_cond")
    city = st.text_input("City (leave blank for any)", key="doc_city")
    fee = st.number_input("Maximum fee in ₹ (0 = any)", 0, 10000, 0, step=100, key="doc_fee")
    if st.button("Recommend doctors", key="doc_btn"):
        if not cond.strip():
            st.warning("Enter a condition first, or run a skin analysis.")
        else:
            from doctor_recommender import recommend, CONDITION_TO_SPECIALTY
            specialty, docs = recommend(cond, city or None, fee or None)
            st.write(f"Suggested specialty: **{specialty}**")
            if not any(k in cond.lower() for k in CONDITION_TO_SPECIALTY):
                st.caption("Condition not recognised, so a General Physician is suggested as a first step.")
            if docs.empty:
                st.info("No available doctors match. Try another city or a higher fee limit.")
            else:
                st.dataframe(
                    docs[["name", "specialty", "city", "rating", "experience_years", "fee"]],
                    hide_index=True)