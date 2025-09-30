import streamlit as st
import pandas as pd
from src.modeling import load_model
import os

st.title("Prédiction")
st.write("Chargez un modèle (`models/model.joblib`) et prédisez.")

if not os.path.exists("models/model.joblib"):
    st.error("Aucun modèle trouvé. Entraînez-en un dans l'onglet Modélisation.")
else:
    pipe = load_model("models/model.joblib")
    uploaded = st.file_uploader("CSV à prédire (sans la cible)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        preds = pipe.predict(df)
        out = df.copy()
        out["prediction"] = preds
        if hasattr(pipe, "predict_proba"):
            out["confidence"] = pipe.predict_proba(df).max(axis=1)
        st.dataframe(out.head(100))
        st.download_button(
            "Télécharger les prédictions (CSV)",
            out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )
