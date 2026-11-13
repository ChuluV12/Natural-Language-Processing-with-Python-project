# Deployment: Researchly Abstract Classifier

A small Streamlit page that labels a pasted abstract as *artificial intelligence* or *classical research*. The model is the class-weighted Linear SVM from notebook 01, saved together with its TF-IDF and cleaning step.

## Files

- `app.py` the web page
- `text_utils.py` the cleaning function (the saved model needs it to load)
- `model.joblib` the trained pipeline
- `requirements.txt` the packages the app needs

## Run it

1. Install Python 3.10 or newer.
2. In this folder run:
   ```
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Open http://localhost:8501 in a browser, paste an abstract, press **Classify Abstract**.

## Notes

- Keep `app.py`, `text_utils.py` and `model.joblib` in the same folder.
- Use the same scikit-learn version that trained the model if loading fails (`model.joblib` was built with scikit-learn 1.9). Rerunning notebook 01 rebuilds the file for your version.
- The model learned from economics and finance abstracts. Its test F1 for the AI class is 0.74, so treat single predictions as a screening aid.
- To publish it, push this folder to GitHub and create an app on Streamlit Community Cloud with `deployment/app.py` as the main file.
