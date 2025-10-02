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

# 1) largeur de page : "centered" ou "wide"
st.set_page_config(layout="centered", page_title="Accidents routiers", page_icon="🚧")

# 2) CSS simple : largeur max + tailles + espacement
st.markdown("""
<style>
/* largeur max du contenu */
.main .block-container {max-width: 1100px; padding-top: 1.5rem; padding-bottom: 4rem;}
/* titres */
h1, h2, h3 { letter-spacing: .2px; }
h1 { font-size: 1.9rem !important; }
h2 { font-size: 1.4rem !important; margin-top: 1.2rem; }
/* paragraphes */
p, li { line-height: 1.6; font-size: 0.98rem; }
/* séparateurs plus discrets */
hr { margin: .8rem 0 1.2rem 0; opacity: .4; }
/* badges */
.badge { display:inline-block; padding: .15rem .5rem; border-radius: .5rem; background:#111827; color:#E5E7EB; font-size:.75rem; }
.card { border:1px solid rgba(255,255,255,.1); border-radius: 12px; padding: .9rem 1rem; margin:.4rem 0; }
</style>
""", unsafe_allow_html=True)

#########

# Petits badges
st.markdown('<span class="badge">Dataset 2005–2018</span> '
            '<span class="badge">Classification</span> '
            '<span class="badge">Streamlit Demo</span>', unsafe_allow_html=True)

st.divider()



########################################
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

    for annee in range(2005, 2019):                                            #### Filtrer jusqu'en 2009 au lieu de 2019 (pour les tests)
        chemin = f'data/sample_caracteristiques_{annee}.csv'  
        #sep = '\t' if annee == 2009 else ','
        sep=','
        df = pd.read_csv(
            chemin, sep=',', encoding='latin1',
            dtype={'dep': str, 'com': str, 'hrmn': str}  
        )
        df['annee'] = annee
        caracteristiques_2005_2018.append(df)

    caracs = pd.concat(caracteristiques_2005_2018, ignore_index=True)
    return caracs




####

# KPIs (ex: après avoir chargé caracs)
c1, c2, c3 = st.columns(3)
c1.metric("Lignes", f"{caracs.shape[0]:,}".replace(",", " "))
c2.metric("Colonnes", caracs.shape[1])
#c3.metric("Mémoire (Mo)", round(caracs.memory_usage(deep=True).sum()/1024**2, 2))

st.divider()

# Onglets pour organiser ton code existant (colle tes blocs EDA dans les bons onglets)
tab1, tab2, tab3 = st.tabs(["📥 Chargement", "🔍 Exploration", "🧼 Nettoyage"])
with tab1:
    st.markdown("#### Chargement des fichiers")
    # ⬇️ colle ici ton bloc de lecture / concat
    # st.dataframe(caracs.head())
    # Appel
    caracs = load_caracteristiques_2005_2018()
    st.dataframe(caracs.head())

with tab2:
    st.markdown("#### Aperçus & distributions")
    # ⬇️ colle ici tes .head(), .info() (version st.code), histos, countplot, etc.
    # info()
    buf = io.StringIO()
    caracs.info(buf=buf)
    s=buf.getvalue()
    st.code(s, language="text")

with tab3:
    st.markdown("#### Nettoyages appliqués")
    # ⬇️ colle ici tes transformations (rename, types, fillna...), puis un aperçu








#############################################################################
##                              Usagers                                    ##
#############################################################################
st.markdown("<u><font size=5>__Usagers__</font></u>", unsafe_allow_html=True)
st.write("")

@st.cache_data
def load_usagers_2005_2018():
    usagers_2005_2018 = []
    for annee in range(2005, 2019):                                            #### Filtrer jusqu'en 2009 au lieu de 2019 (pour les tests)
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
    for annee in range(2005, 2019):                                            #### Filtrer jusqu'en 2009 au lieu de 2019 (pour les tests)
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
    for annee in range(2005, 2019):                                            #### Filtrer jusqu'en 2009 au lieu de 2019 (pour les tests)
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

#####
## Nombre d'accident par années (=>sample => 1000 lignes = 1000 accidents)
#####

# Compter les accidents par année
caracs.reset_index(drop=True, inplace=True)
accidents_par_annee = caracs['annee'].value_counts().sort_index()

# Liste complète des années (même si certaines années ont 0 accidents)
annees = list(range(2005, 2019))                                            #### Filtrer jusqu'en 2009 au lieu de 2019 (pour les tests)

# S'assurer que toutes les années sont présentes, même avec 0
accidents_par_annee = accidents_par_annee.reindex(annees, fill_value=0)

# Tracer la courbe
plt.figure(figsize=(6, 4))
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
annees = list(range(2005, 2019))                                            #### Filtrer jusqu'en 2009 au lieu de 2019 (pour les tests)
accidents_par_annee = accidents_par_annee.reindex(annees, fill_value=0)
fig = px.line(
    x=accidents_par_annee.index,
    y=accidents_par_annee.values,
    markers=True,
    labels={"x": "Année", "y": "Nombre d'accidents"},
    title="Évolution du nombre d'accidents par année"
)
st.plotly_chart(fig, use_container_width=True)

######
##  Nombre d'accidents par mois
######
# Noms des mois en français
noms_mois = ['janvier', 'février', 'mars', 'avril', 'mai', 'juin',
             'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre']

# Compter les accidents par mois
accidents_par_mois = caracs['mois'].value_counts().sort_index()

# Tracé avec palette et suppression du warning via hue
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=noms_mois, y=accidents_par_mois.values, hue=noms_mois, palette="Greens_d", legend=False)

plt.title("Nombre d'accidents par mois")
plt.xlabel("Mois")
plt.ylabel("Nombre d'accidents")

# Ajouter plus de ticks sur l'axe Y
max_y = accidents_par_mois.max()
plt.yticks(range(0, max_y + 500, 500))                                            #### Filtrer jusqu'en 2009 au lieu de 2019 (pour les tests)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(plt.gcf())
