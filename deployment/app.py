import os

import joblib
import streamlit as st

import text_utils  # noqa: F401  (the saved pipeline needs this module to load)

HERE = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_model():
    bundle = joblib.load(os.path.join(HERE, "model.joblib"))
    return bundle["pipeline"], bundle["labels"]


st.set_page_config(page_title="Researchly Abstract Classifier")
st.title("Researchly Research Abstract Classifier")
st.write("This app classifies research abstracts as either AI-related or classical research.")

abstract = st.text_area("Enter research abstract:", height=200)

if st.button("Classify Abstract"):
    if len(abstract.split()) < 5:
        st.warning("Please enter an abstract of at least a few words.")
    else:
        pipeline, labels = load_model()
        pred = int(pipeline.predict([abstract])[0])
        st.subheader("Prediction")
        st.success(f"The abstract is classified as: {labels[pred]}")
