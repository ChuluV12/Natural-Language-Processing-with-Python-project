# Researchly text mining project

Text mining coursework for the Researchly (SAnalytics) case: classify research abstracts as AI-related or classical, then mine a small corpus of articles for summaries and topics.

## Contents

| Path | What it holds |
|---|---|
| `notebooks/01_abstract_classification.ipynb` | Questions 1 and 2: cleaning, feature choice, baseline, four models, class weighting, saved model |
| `notebooks/02_corpus_preprocessing_and_summarisation.ipynb` | Question 3: corpus dataframe, denoising, two summaries |
| `notebooks/03_topic_modelling_lda.ipynb` | Question 4: LDA topics and the pyLDAvis map |
| `data/research_abstracts.json` | 1,138 labelled abstracts |
| `data/research_corpus/` | five article text files |
| `deployment/` | Streamlit web app, saved model and its own README |
| `outputs/` | figures, cleaned corpus, LDA topic words, `lda_intertopic_map.html` |
| `docs/Researchly_Project_Report.pdf` | written documentation for Questions 1 to 4 |

## Running the notebooks

```
pip install -r requirements.txt
cd notebooks
jupyter notebook
```

Run them in order. Notebook 01 writes `deployment/model.joblib`, notebook 02 writes `outputs/corpus_clean.csv`, and notebook 03 reads that file. All random seeds are fixed at 42. No NLTK downloads are needed; stop words come from scikit-learn and lemmas from `simplemma`.

## Running the web app

See `deployment/README.md`. Short version:

```
cd deployment
pip install -r requirements.txt
streamlit run app.py
```

## Results in one paragraph

On a stratified 20% hold-out set, the Multinomial Naive Bayes baseline reaches F1 0.695 for the AI class. A Linear SVM with class weighting, selected by cross-validation, reaches 0.736. Over three repeats of 5-fold cross-validation the gain is 0.745 to 0.780 F1. LDA on the five-article corpus finds four coherent topics (c_v 0.554).
