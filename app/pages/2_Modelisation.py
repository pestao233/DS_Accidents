import streamlit as st
import pandas as pd
from src.data_processing import split_xy, make_preprocess, train_val_split
from src.modeling import make_models, train_with_smote, evaluate, save_model
import os

st.title("Modélisation")
st.write("**But :** entraîner un modèle baseline et sauvegarder `models/model.joblib`.")

uploaded = st.file_uploader("CSV complet (avec cible)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    target = st.selectbox("Sélectionner la cible", options=df.columns, index=len(df.columns)-1)
    X, y = split_xy(df, target)
    X_train, X_test, y_train, y_test = train_val_split(X, y)
    pre = make_preprocess(X_train)

    algo = st.selectbox("Algorithme", ["rf", "hgb"], format_func=lambda k: {"rf": "RandomForest", "hgb":"HistGradientBoosting"}[k])
    models = make_models()
    if st.button("Entraîner le modèle"):
        pipe = train_with_smote(pre, models[algo], X_train, y_train)
        res = evaluate(pipe, X_test, y_test)
        st.json(res["report"])
        os.makedirs("models", exist_ok=True)
        save_model(pipe, "models/model.joblib")
        st.success("Modèle sauvegardé → `models/model.joblib`")
else:
    st.info("Chargez un CSV pour entraîner un modèle.")
