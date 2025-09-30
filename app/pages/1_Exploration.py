import streamlit as st

st.markdown("<u>__Cadre__</u>", unsafe_allow_html=True)
st.write("""
Les données proviennent de la base publique des accidents corporels de la circulation en France, disponibles via data.gouv.fr. 
Nous avons utilisé les fichiers annuels entre 2005 et 2018, structurés en quatre tables :
- caracteristiques : infos principales sur l’accident (date, heure, luminosité, lieu, météo, etc.)
- lieux : configuration de la route et conditions d'infrastructure
- vehicules : type de véhicules impliqués et point d'impact
- usagers : profil et gravité des personnes impliquées
Une difficulté notable a été la gestion de l’hétérogénéité entre fichiers annuels : encodages variables (latin1 ou utf-8), séparateurs différents (, ou \t), colonnes présentes ou absentes selon l'année.
Un script de lecture automatique a été mis en place pour charger les 4 tables par année en homogénéisant les formats.
La volumétrie est importante : chaque fichier annuel contient plusieurs dizaines de milliers de lignes, pour un total d’environ 1 million d’accidents sur la période étudiée.
Nous avons aussi contrôlé la cohérence des identifiants Num_Acc entre les tables et aucune différence identifiée.
Nous avons également dans ces fichiers beaucoup de variables qui ne sont pas pertinentes pour notre étude.
Il a donc fallu évaluer leur pertinence afin de décider de les garder ou non.
""")
    
import pandas as pd
import plotly.express as px

st.title("Exploration des données")
st.write("Chargements des dataframes")

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
