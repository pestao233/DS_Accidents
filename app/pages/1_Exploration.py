import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import chi2_contingency
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
import io


st.title("Exploration des données")

st.markdown("<u><font size=5>__Cadre__</u>", unsafe_allow_html=True)
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

#############################################################################
##                      Caractéristiques                                   ##
#############################################################################

st.markdown("<u><font size=5>__Caractéristiques__</font></u>", unsafe_allow_html=True)
st.write("")

@st.cache_data
def load_caracteristiques_2005_2018():
    caracteristiques_2005_2018 = []
    dtypes_caracs = {
        "dep": "category", "com": "category",
        "lum": "int8", "agg": "int8", "int": "int8",
        "atm": "int8", "col": "int8",
        "mois": "int8", "jour": "int8", "an": "int16"
    }

    for annee in range(2005, 2009):
        chemin = f'data/sample_caracteristiques_{annee}.csv'  
        sep = '\t' if annee == 2009 else ','
        df = pd.read_csv(
            chemin, sep=sep, encoding='latin1',
            dtype={'dep': str, 'com': str, 'hrmn': str}  
        )
        df['annee'] = annee
        caracteristiques_2005_2018.append(df)

    caracs = pd.concat(caracteristiques_2005_2018, ignore_index=True)
    return caracs

# Appel
caracs = load_caracteristiques_2005_2018()
st.dataframe(caracs.head())

# info()
buf = io.StringIO()
caracs.info(buf=buf)
s=buf.getvalue()
st.code(s, language="text")

#############################################################################
##                              Usagers                                    ##
#############################################################################
st.markdown("<u><font size=5>__Usagers__</font></u>", unsafe_allow_html=True)
st.write("")

@st.cache_data
def load_usagers_2005_2018():
    usagers_2005_2018 = []
    for annee in range(2005, 2009):
        chemin = f'data/sample_usagers_{annee}.csv'
        df = pd.read_csv(chemin, sep=',', encoding='latin1')
        df['annee'] = annee
        usagers_2005_2018.append(df)

    usagers = pd.concat(usagers_2005_2018, ignore_index=True)
    return usagers

# Appel
usagers = load_usagers_2005_2018()
st.dataframe(usagers.head())

# info()
buf = io.StringIO()
usagers.info(buf=buf)
s=buf.getvalue()
st.code(s, language="text")

#############################################################################
##                              Lieux                                      ##
#############################################################################
st.markdown("<u><font size=5>__Lieux__</font></u>", unsafe_allow_html=True)
st.write("")

@st.cache_data
def load_lieux_2005_2018():
    lieux_2005_2018 = []
    for annee in range(2005, 2009):
        chemin = f'data/sample_lieux_{annee}.csv'
        df = pd.read_csv(chemin, sep=',', encoding='latin1')
        df['annee'] = annee
        lieux_2005_2018.append(df)

    lieux = pd.concat(lieux_2005_2018, ignore_index=True)
    return lieux

# Appel
lieux = load_lieux_2005_2018()
st.dataframe(lieux.head())

# info()
buf = io.StringIO()
lieux.info(buf=buf)
s=buf.getvalue()
st.code(s, language="text")

#############################################################################
##                              Vehicules                                  ##
#############################################################################
st.markdown("<u><font size=5>__Vehicules__</font></u>", unsafe_allow_html=True)
st.write("")

@st.cache_data
def load_vehicules_2005_2018():
    vehicules_2005_2018 = []
    for annee in range(2005, 2009):
        chemin = f'data/sample_vehicules_{annee}.csv'
        df = pd.read_csv(chemin, sep=',', encoding='latin1')
        df['annee'] = annee
        vehicules_2005_2018.append(df)

    vehicules = pd.concat(vehicules_2005_2018, ignore_index=True)
    return vehicules

# Appel
vehicules = load_vehicules_2005_2018()
st.dataframe(vehicules.head())

# info()
buf = io.StringIO()
vehicules.info(buf=buf)
s=buf.getvalue()
st.code(s, language="text")


#############################################################################
##                                DataViz                                  ##
#############################################################################
## Nombre d'accident par années

# Compter les accidents par année
accidents_par_annee = caracs['annee'].value_counts().sort_index()

# Liste complète des années (même si certaines années ont 0 accidents)
annees = list(range(2005, 2009))

# S'assurer que toutes les années sont présentes, même avec 0
accidents_par_annee = accidents_par_annee.reindex(annees, fill_value=0)

# Tracer la courbe
plt.figure(figsize=(7, 4))
sns.lineplot(x=accidents_par_annee.index, y=accidents_par_annee.values, marker='o')
plt.title("Évolution du nombre d'accidents par année")
plt.xlabel("Année")
plt.ylabel("Nombre d'accidents")
plt.xticks(annees, rotation=45)
plt.tight_layout()
st.pyplot(plt.gcf())

# option 2
import plotly.express as px
accidents_par_annee = caracs['annee'].value_counts().sort_index()
annees = list(range(2005, 2009))
accidents_par_annee = accidents_par_annee.reindex(annees, fill_value=0)
fig = px.line(
    x=accidents_par_annee.index,
    y=accidents_par_annee.values,
    markers=True,
    labels={"x": "Année", "y": "Nombre d'accidents"},
    title="Évolution du nombre d'accidents par année"
)
st.plotly_chart(fig, use_container_width=True)

