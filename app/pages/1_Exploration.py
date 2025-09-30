import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Exploration des données")
st.write("Chargez un échantillon CSV pour l'exploration rapide.")

uploaded = st.file_uploader("CSV (<= 50 Mo)", type=["csv"])

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

if uploaded:
    df = load_csv(uploaded)
    st.success(f"Shape: {df.shape}")
    st.dataframe(df.head(50))
    st.subheader("Types / Valeurs manquantes")
    st.write(df.dtypes)
    st.write(df.isna().mean().sort_values(ascending=False).head(20).to_frame("taux_na"))
    target = st.selectbox("Cible (target)", options=df.columns, index=len(df.columns)-1)
    st.subheader("Distribution de la cible")
    if df[target].nunique() < 50:
        st.plotly_chart(px.histogram(df, x=target), use_container_width=True)
    else:
        st.warning("Cible à forte cardinalité, histogramme non affiché.")
else:
    st.info("Déposez un CSV pour commencer.")
