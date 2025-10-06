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
from PIL import Image

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
- **caracteristiques** : infos principales sur l’accident (date, heure, luminosité, lieu, météo, etc.)
- **lieux** : configuration de la route et conditions d'infrastructure
- **vehicules** : type de véhicules impliqués et point d'impact
- **usagers** : profil et gravité des personnes impliquées\n
Une difficulté notable a été la gestion de l’hétérogénéité entre fichiers annuels : encodages variables (latin1 ou utf-8), séparateurs différents (, ou \t), colonnes présentes ou absentes selon l'année.

Un script de lecture automatique a été mis en place pour charger les 4 tables par année en homogénéisant les formats.\n
La volumétrie est importante : chaque fichier annuel contient plusieurs dizaines de milliers de lignes, pour un total d’environ 1 million d’accidents sur la période étudiée.\n
Nous avons aussi contrôlé la cohérence des identifiants Num_Acc entre les tables et aucune différence identifiée.\n
Nous avons également dans ces fichiers beaucoup de variables qui ne sont pas pertinentes pour notre étude.
Il a donc fallu évaluer leur pertinence afin de décider de les garder ou non.
""")

#############################################################################
##                      Caractéristiques                                   ##
#############################################################################

#st.markdown("<u><font size=5>__Caractéristiques__</font></u>", unsafe_allow_html=True)

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

# Appel
caracs = load_caracteristiques_2005_2018()


#############################################################################
##                              Usagers                                    ##
#############################################################################
#st.markdown("<u><font size=5>__Usagers__</font></u>", unsafe_allow_html=True)

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



#############################################################################
##                              Lieux                                      ##
#############################################################################
#st.markdown("<u><font size=5>__Lieux__</font></u>", unsafe_allow_html=True)

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


#############################################################################
##                              Vehicules                                  ##
#############################################################################
#st.markdown("<u><font size=5>__Vehicules__</font></u>", unsafe_allow_html=True)

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


#############################################################################
##                                Head/info                                ##
#############################################################################

st.divider()

# Onglets pour organiser ton code existant (colle tes blocs EDA dans les bons onglets)
tab1, tab2, tab3 = st.tabs(["📥 Chargement", "🔍 Exploration / 🧼 Nettoyage", "📊 Dataviz"])
with tab1:
    st.markdown("#### Aperçu du DataFrame : `Caractéristiques`")
    st.dataframe(caracs.head())
    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes totales (full):", "958 469")
    c2.metric("Lignes totales (sample):", f"{caracs.shape[0]:,}".replace(",", " "))
    c3.metric("Colonnes totales:", caracs.shape[1])
    #c3.metric("Mémoire (Mo)", round(caracs.memory_usage(deep=True).sum()/1024**2, 2))

    st.subheader("Résumé du DataFrame : `Caractéristiques`")
    st.code("""
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 958469 entries, 0 to 958468
    Data columns (total 17 columns):
    #   Column   Non-Null Count   Dtype  
    ---  ------   --------------   -----  
    0   Num_Acc  958469 non-null  int64  
    1   an       958469 non-null  int64  
    2   mois     958469 non-null  int64  
    3   jour     958469 non-null  int64  
    4   hrmn     958469 non-null  object 
    5   lum      958469 non-null  int64  
    6   agg      958469 non-null  int64  
    7   int      958469 non-null  int64  
    8   atm      958396 non-null  float64
    9   col      958450 non-null  float64
    10  com      958467 non-null  object 
    11  adr      816550 non-null  object 
    12  gps      480052 non-null  object 
    13  lat      471401 non-null  float64
    14  long     471397 non-null  object 
    15  dep      958469 non-null  object 
    16  annee    958469 non-null  int64  
    dtypes: float64(3), int64(8), object(6)
    """)

    st.divider()

    st.markdown("#### Aperçu du DataFrame : `Usagers`")
    st.dataframe(usagers.head())
    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes totales (full):", "2 142 195")
    c2.metric("Lignes totales (sample):", f"{usagers.shape[0]:,}".replace(",", " "))
    c3.metric("Colonnes totales:", usagers.shape[1])
    #c3.metric("Mémoire (Mo)", round(usagers.memory_usage(deep=True).sum()/1024**2, 2))
   
    st.subheader("Résumé du DataFrame : `Usagers`")
    st.code("""
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2142195 entries, 0 to 2142194
    Data columns (total 13 columns):
    #   Column   Dtype  
    ---  ------   -----  
    0   Num_Acc  int64  
    1   place    float64
    2   catu     int64  
    3   grav     int64  
    4   sexe     int64  
    5   trajet   float64
    6   secu     float64
    7   locp     float64
    8   actp     float64
    9   etatp    float64
    10  an_nais  float64
    11  num_veh  object 
    12  annee    int64  
    dtypes: float64(7), int64(5), object(1)
    """)
    
    st.divider()

    st.markdown("#### Aperçu du DataFrame : `Lieux`")
    st.dataframe(lieux.head())
    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes totales (full):", "958 469")
    c2.metric("Lignes totales (sample):", f"{lieux.shape[0]:,}".replace(",", " "))
    c3.metric("Colonnes totales:", lieux.shape[1])
    #c3.metric("Mémoire (Mo)", round(lieux.memory_usage(deep=True).sum()/1024**2, 2))
            
    st.subheader("Résumé du DataFrame : `Lieux`")
    st.code("""
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 958469 entries, 0 to 958468
    Data columns (total 19 columns):
    #   Column   Non-Null Count   Dtype  
    ---  ------   --------------   -----  
    0   Num_Acc  958469 non-null  int64  
    1   catr     958468 non-null  float64
    2   voie     869558 non-null  object 
    3   v1       333391 non-null  float64
    4   v2       39348 non-null   object 
    5   circ     956895 non-null  float64
    6   nbv      955738 non-null  float64
    7   pr       482985 non-null  float64
    8   pr1      481166 non-null  float64
    9   vosp     955708 non-null  float64
    10  prof     956520 non-null  float64
    11  plan     956188 non-null  float64
    12  lartpc   902271 non-null  float64
    13  larrout  904096 non-null  float64
    14  surf     956545 non-null  float64
    15  infra    953061 non-null  float64
    16  situ     953499 non-null  float64
    17  env1     953029 non-null  float64
    18  annee    958469 non-null  int64  
    dtypes: float64(15), int64(2), object(2)
    """)
    st.divider()

    st.markdown("#### Aperçu du DataFrame : `Véhicules`")
    st.dataframe(vehicules.head())
    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes totales (full):", "1 635 811")
    c2.metric("Lignes totales (sample):", f"{vehicules.shape[0]:,}".replace(",", " "))
    c3.metric("Colonnes totales:", vehicules.shape[1])
    #c3.metric("Mémoire (Mo)", round(vehicules.memory_usage(deep=True).sum()/1024**2, 2))  

    st.subheader("Résumé du DataFrame `Vehicules`")
    st.code("""
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1635811 entries, 0 to 1635810
    Data columns (total 10 columns):
    #   Column   Non-Null Count    Dtype  
    ---  ------   --------------    -----  
    0   Num_Acc  1635811 non-null  int64  
    1   senc     1635539 non-null  float64
    2   catv     1635811 non-null  int64  
    3   occutc   1635811 non-null  int64  
    4   obs      1634805 non-null  float64
    5   obsm     1635033 non-null  float64
    6   choc     1635414 non-null  float64
    7   manv     1635343 non-null  float64
    8   num_veh  1635811 non-null  object 
    9   annee    1635811 non-null  int64  
    dtypes: float64(5), int64(4), object(1)
    """)

            
with tab2:
    st.write("### Préparation des données")
    st.write(""" Pour préparer au mieux les données en vue de la modélisation, nous avons testé deux approches :
    - la première consistait à prétraiter chaque DataFrame séparément avant de les fusionner,
    - La seconde visait à fusionner l’ensemble des fichiers bruts avant d’appliquer un prétraitement global.
    Les deux méthodes ont conduit à des résultats relativement similaires. 
    
    Cependant, afin de limiter la perte d’informations et de lignes d’accidents lors du traitement post-fusion, nous avons retenu la seconde approche : fusionner puis traiter.
    """)

    st.code("""
    # On merge les différents DataFrame en un seul                
    df_merge_accident = caracs
        .merge(vehicules, on='Num_Acc', how='left', suffixes=('', '_veh')) 
        .merge(usagers, on='Num_Acc', how='left', suffixes=('', '_usag')) 
        .merge(lieux, on='Num_Acc', how='left', suffixes=('', '_lieu'))
    """, language="python")

    st.write("### Gestion des valeurs manquantes et des doublons")
    st.write("""Le premier défi auquel nous avons été confronté lors du pré-traitement a été la **gestion des valeurs manquantes et des doublons**.\n""")
    st.write("""Ces doublons provenaient du fait qu’un même accident pouvait apparaître plusieurs fois dans le jeu de données, notamment lorsqu’il était décrit au niveau des **véhicules** ou des **usagers**.\n""")
    st.write("""Comme notre objectif était d’analyser la gravité de l’accident, nous avons réencodé et agrégé certaines variables afin d’obtenir un jeu de données unique, contenant **une seule ligne par accident**.\n
 """)
    st.write("""Par ailleurs, les accidents comportent un grand nombre de facteurs — tels que le lieu, la présence éventuelle de piétons, l’équipement de sécurité, ou encore le type de véhicule — ce qui a entraîné la présence de **nombreuses valeurs manquantes dans certaines variables**.      
    """)
    
    st.write("### Gestion des Nans")
    st.write("""Le DataFrame final comportait plusieurs colonnes inutiles pour notre analyse que nous avons supprimés directement""")
    
    st.code("""
    # On supprime les colonnes inutiles               
    df_merge_accident = df_merge_accident.drop(['voie', 'annee','adr', 'gps', 'lat', 'long', 'vosp', 'v1', 'v2', 'pr', 'pr1', 'plan', 'env1'], axis = 1)
    """, language="python")

    st.code("""
    # On supprime les nans lorsqu'il y en a très peu             
    df_merge_accident = df_merge_accident.dropna(subset=['atm', 'col', 'com', 'catr', 'choc', 'manv', 'senc'])
    """, language="python")

    st.write("""Afin d’éviter les biais liés aux valeurs manquantes, nous avons appliqué un **remplissage différencié selon le type de variable** :
    les champs catégoriels inconnus ont été codés par **-1**, les valeurs binaires absentes par **0**, et certaines variables numériques par leur **valeur médiane**.
            """)

    with st.expander("Afficher le code"):
        st.code("""
        ## On remplace les Nans de obs et obm par -1
        df_merge_accident[['obs', 'obsm']] = df_merge_accident[['obs', 'obsm']].fillna(-1)
            
        # lartpc : NaN = pas de TPC
        df_merge_accident['lartpc'] = df_merge_accident['lartpc'].fillna(0)

        # larrout : NaN = inconnu / non mesuré
        df_merge_accident['larrout'] = df_merge_accident['larrout'].fillna(-1)

        # prof : inconnu
        df_merge_accident['prof'] = df_merge_accident['prof'].fillna(-1)

        # nbv : médiane 
        df_merge_accident['nbv'] = df_merge_accident['nbv'].fillna(df_merge_accident['nbv'].median())

        # circ : inconnu
        df_merge_accident['circ'] = df_merge_accident['circ'].fillna(-1)

        # infra : NaN = aucun aménagement particulier
        df_merge_accident['infra'] = df_merge_accident['infra'].fillna(0)

        # situ : inconnu
        df_merge_accident['situ'] = df_merge_accident['situ'].fillna(-1)""", language="python")

    st.write("""Afin de mieux pouvoir utiliser certaines variables, mais également afin de supprimer les doublons, et de n'obtenir qu'une ligne par accident, nous avons réencodé plusieurs variables""")

    st.write("### Transformation des variables 'sécurité', et 'num_veh'")
    st.write("""Nous avons **fusionné les informations de la variable sécurité**, qui distinguait la présence et l’utilisation d’un équipement à travers une valeur numérique, afin d’obtenir **une seule variable** indiquant simplement **si un équipement a été utilisé ou non**.""")
    st.write("""De plus, la colonne **num_veh a été supprimée** et remplacée par une variable correspondant au **nombre de véhicules impliqués dans chaque accident**.
    """)
    with st.expander("Exemple code variable sécurité"):
        st.code("""
        #Si équipement utilisé, on l'indique. Si pas utilisé on met 0, si non déterminable, on met 3.
        def decode_secu(val):
            if pd.isna(val):
                return -1
            val = int(val)
            equipement = val // 10
            usage = val % 10
            if usage == 1:
                return equipement
            elif usage == 2:
                return 0
            elif usage == 3:
                return -1
            else:
                return -1  # cas imprévu
        df_merge_accident['secu'] = df_merge_accident['secu'].apply(decode_secu).astype('Int64')
            """, language="python")


    st.write("### Réencodage")
    st.write("""Afin de supprimer les doublons, et de n'obtenir qu'une ligne par accident, nous avont réencoder plusieurs variables.""")

    st.write("""Nous avons créé plusieurs familles afin de regrouper les variables et avons créé un OneHotEncoder lorsque nécessaire""")

    st.write("""Il était parfois necessaire d'effectuer des tests de Chi et Cramer pour comprendre l'importance des variables par rapport à la variable cible.
    Nous avons également dû analyser leur répartition pour comprendre comment les encoder""")

    with st.expander("Exemple de code pour la variable choc"):
        st.code("""
        ##Colonne choc, on réduit le nombre de colonnes 
        ## Créer les 4 familles directement
        df_merge_accident['choc_avant']    = df_merge_accident['choc'].isin([1,2,3]).astype(int)
        df_merge_accident['choc_arriere']  = df_merge_accident['choc'].isin([4,5,6]).astype(int)
        df_merge_accident['choc_lateral']  = df_merge_accident['choc'].isin([7,8]).astype(int)
        df_merge_accident['choc_multiple'] = (df_merge_accident['choc'] == 9).astype(int)

        ## Agréger pour avoir une ligne par accident
        choc_flags = (df_merge_accident.groupby('Num_Acc', as_index=False)
                        [['choc_avant','choc_arriere','choc_lateral','choc_multiple']]
                        .max())
        """, language="python")

    st.write("###### Variable surf")
    st.write("""
    - Normal : 1 
    - Défavorable : 0   
    - Inconnu : -1""")

    st.write("###### Variable Sexe")
    st.write("""On a divisé la colonne sexe en deux colonnes : Homme et Femme""")

    st.write("###### Variable place")
    st.write("""
    - Conducteur : place 1
    - Passager avant : place 2
    - Passager arrière : places 3 à 6
    - Autres / indéterminé : places 7, 8, 9, 0""")

    st.write("###### Variable Trajet")
    st.write("""
    - 0: "Inconnu"
    - 1: "Travail"        
    - 4: "Professionnel"   
    - 5: "Loisirs"          
    - 2, 3, 9: "Autre"
            """)          

    st.write("###### Variable âge")
    st.write("""
    Nous avons vérifié si les valeurs aberrantes provenaient de fautes de frappe pouvant être corrigées, ce qui n’était pas le cas.

    Comme avec les Nans, elles représentaient **moins de 1%** des données, nous avons choisi de les éliminer, en les remplaçant par la médiane ±5.
    Enfin, nous avons regroupé les individus par classes d’âge afin de faciliter l’analyse.
    - age < 18: "Enfant"
    - age < 30: "Jeune"
    - age < 60: "Adulte"
    - age > 60: "Senior"
            """)

    st.write("### La variable cible : Variable gravité :")
    st.write("""Variable gravité
    La variable gravité est la variable que nous souhaitons prédire, nous l'avons réparti en 4 classes :
    - 1:"Indemne" 
    - 2:"Tué"
    - 3:"Blessé hospitalisé"
    - 4:"Blessé léger" 
    Nous avons ensuite créée une variable, en sélectionnant la gravité maximale de l'accident  """)

    st.code("""label_grav = {1:"Indemne", 2:"Tué", 3:"Blessé hospitalisé", 4:"Blessé léger"}
    ordre_gravite = {1:0, 4:1, 3:2, 2:3}  # échelle: indemne < léger < hosp < tué

    u = df_merge_accident.copy()

    # Colonnes indicatrices par usager
    u['tue']   = (u['grav'] == 2).astype(int)
    u['hosp']  = (u['grav'] == 3).astype(int)
    u['leger'] = (u['grav'] == 4).astype(int)
    u['indem'] = (u['grav'] == 1).astype(int)

    # Gravité max
    u['grav_order'] = u['grav'].map(ordre_gravite)

    accident = u.groupby('Num_Acc').agg(grav_order_max = ('grav_order', 'max')).reset_index()""", language="python")

    st.write("### Traitement des variables catégorielles et réduction de la colinéarité :")
    st.write("""Lorsque des OneHotEncoding avait été appliqué, nous avons supprimé une modalité de référence pour chaque variable afin d’éviter la colinéarité.
    Les colonnes redondantes ou apportant des informations similaires ont également été retirées pour simplifier et stabiliser le jeu de données.
    """)

    st.write("###### Variable hrmn")                  
    st.write("""Nous avons encodé hrmn en deux variables "heure" et "minute".      
    Nous avons supprimé minute.""")

    st.write("### Train/test split")
    st.code("""##On split
    from sklearn.model_selection import train_test_split
    X = df_fin.drop(columns=["grav_order_max", "Num_Acc", "an", "minute"])
    y = df_fin["grav_order_max"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    """, language="python")

    st.write("### Encodage des variables catégorielles")
    st.write("""Les colonnes heure, jour et mois créant probablement une mauvais information, nous les avons traités comme des variables catégorielles
    """)
    st.code("""cat_cols = ["mois", "jour", "heure"]
    num_cols = X.columns.difference(cat_cols)

    # Prétraitement catégorielles (One-Hot)
    cat_tf = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", drop="first"))])

    # ColumnTransformer
    preprocess = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", cat_tf, cat_cols),
    ])
            
    # Fit uniquement sur le train
    preprocess.fit(X_train)

    # Transforme train et test
    X_train_transformed = preprocess.transform(X_train)
    X_test_transformed = preprocess.transform(X_test)
            """, language="python")

    st.write("### Standardisation et rééquilibrage des classes")
    st.write("""Nous avons remarqué que nous avions un déséquilibre des classes, que nous avons traité 
    """)

    st.write("""Comme nous avons rencontré à plusieurs reprise des problèmes de mémoire, nous avons d'abord opté pour un undersampling léger, puis ensuite un SMOTE afin de rééquilibrer 
    """)

    st.write("#### Standardisation ")
    st.code("""
    #Standardisation

    scaler = StandardScaler(with_mean=False)
    scaler.fit(X_train_transformed)

    X_train_scaled = scaler.transform(X_train_transformed)
    X_test_scaled = scaler.transform(X_test_transformed)
    """, language="python")

    st.write("#### Oversampling et Undersampling ")
    st.code("""
    from collections import Counter
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import SMOTE

    # Undersampling léger
    # On réduit un peu la classe majoritaire sans tomber au niveau des minoritaires
    cnt = Counter(y_train)
    strategy_under = {
        2: int(cnt[2] * 0.5),  
        3: int(cnt[3] * 0.8),  
        4: cnt[4]              
    }

    rus = RandomUnderSampler(sampling_strategy=strategy_under, random_state=42)
    X_train_under, y_train_under = rus.fit_resample(X_train_scaled, y_train)
    print("Après undersampling :", Counter(y_train_under))

    # Oversampling (SMOTE) pour équilibrer
    smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=3)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_under, y_train_under)
    """, language="python")



with tab3:
    st.markdown("#### Dataviz")
    #############################################################################
    ##                                DataViz                                  ##
    #############################################################################
            
    #####
    ## Nombre d'accident par années (=>sample => 1000 lignes = 1000 accidents)
    #####
            
    # Chargement de graphes issus du notebook
    #for name in ["Evol_acc_annee.png","Evol_acc_mois.png","Repart_moment.png", "Repart_age.png", "Repart_sexe.png", "Repart_Trajet.png", "Repart_Gravité.png","Dist_agg_grav.png", "Dist_int_grav.png", "Dist_lum_grav.png", "Dist_secu_grav.png", "Corr_var_expl_Grav.png", "Corr_var_expl.png"]
     
    col = st.columns([1,12,1])[1]
    with col:
         # Evol accidents/année
         name="Evol_acc_annee.png"
         st.image(f"reports/{name}", width=400, use_container_width=True)
         st.markdown("""
         **Analyse :** 
         - Baisse nette et continue entre 2005 et 2018 - coïncide avec l'installation de plus de 500 radars en 2005
         """)
         st.divider()

    # Evol accidents/mois
    with col:
         name="Evol_acc_mois.png"
         st.image(f"reports/{name}", width=400, use_container_width=True)
         st.markdown("""
         **Analyse :** 
         - Accidents plus fréquents entre avril et juillet : pic estival lié à l'augmentation des déplacements (vacances, loisirs, + motards/cyclistes)  
         - Niveaux plus faibles en hiver (moins de trajets)
         """)
         st.divider()

    # Repart/moment
    with col:
         name="Repart_moment.png"
         st.image(f"reports/{name}", width=400, use_container_width=True)
         st.markdown("""
         **Analyse :** 
         - L'après-midi est la période la plus accidentogène (38.5%) : corrélé à un trafic élevé et à un rythme de circulation actif  
         - Le soir suit en intensité (baisse de luminosité + fatigue accrue)
         - Le matin a une contribution modérée  
         - La nuit (0h-6h) est la moins fréquentée, mais les accidents qui y surviennent peuvent être plus graves  
         """)
         st.divider()

    # Repart/age
    with col:
         name="Repart_age.png"
         st.image(f"reports/{name}", width=400, use_container_width=True)
         st.markdown("""
         **Analyse :** 
         - Les **adultes** constituent la majorité des usagers accidentés, en lien avec leur exposition plus forte au trafic quotidien
         - Les **jeunes** représentent une part importante : comportements à risque et expérience moindre 
         - Les **seniors** et **enfants** sont moins présents en nombre, mais plus vulnérables en cas d'accident
         """)
         st.divider()
     
    # Repart/sexe
    with col:
         name="Repart_sexe.png"
         st.image(f"reports/{name}", width=400, use_container_width=True)
         st.markdown("""
         **Analyse :** 
         - Les **hommes** représentent environ deux tiers des usagers impliqués dans un accident
         - Cette surreprésentation peut s'expliquer par une plus forte exposition et des comportements à risque plus fréquent (vitesse, alcool, ..)
         - Les **femmes** apparaissent moins impliquées, ce qui traduit à la fois une exposition moindre et des comportements plus prudents
         """)
         st.divider()
     
    # Repart/Trajet
    with col:
         name="Repart_Trajet.png"
         st.image(f"reports/{name}", width=400, use_container_width=True)
         st.markdown("""
         **Analyse :** 
         - Les **trajets de loisirs** sont majoritaires parmi les usagers impliqués, 
          reflétant une **plus forte exposition au risque** lors des déplacements non professionnels
         - Les **trajets domicile-travail** restent fréquents mais sur des distances plus courtes
         - Les **trajets professionnels** sont minoritaires, bien que potentiellement plus graves
         - Une part importante de **trajets "inconnus"** correspond à des données non renseignées
         """)
         st.divider()
     
    # Repart/Gravité
    with col:
         name="Repart_Gravité.png"
         st.image(f"reports/{name}", width=400, use_container_width=True)
         st.markdown("""
         **Analyse :** 
         - La majorité des accidents impliquent des **usagers indemnes**
         - Les **blessés légers (incl. hospitalisés)** représentent environ un tiers des cas
         - Les **accidents mortels** sont minoritaires
         - Cette distribution est déséquilibrée : ce sera un **enjeu clé pour la modélisation** (classes minoritaires)
         """)
         st.divider()

    # Distribution agg/gravite
    with col:
         name="Dist_agg_grav.png"
         st.image(f"reports/{name}", width=400, use_container_width=True)
         st.markdown("""
         **Analyse :** 
         - Les **accidents hors agglomération** sont proportionnellement **plus graves** (davantage de blessés graves et de tués)
         - En **agglomération**, les accidents sont **plus fréquents mais souvent moins violents**
         - Ce contraste met en évidence le rôle de la **vitesse** et des **infrastructures** sur la gravité
         """)
         st.divider()
     
    # Distribution int/gravite
    with col:
         name="Dist_int_grav.png"
         st.image(f"reports/{name}", width=400, use_container_width=True)
         with st.expander("Définition des labels `int` (type d'intersection)"):
             st.markdown("""
             | Code | Type d'intersection |
             |------|----------------------|
             | 0 | Hors intersection |
             | 1 | Intersection en X |
             | 2 | Intersection en T |
             | 3 | Intersection en Y |
             | 4 | Carrefour à sens giratoire |
             | 5 | Place |
             | 6 | Passage à niveau |
             | 7 | Autre intersection |
             | 8 | Échangeur ou bretelle d'autoroute |
             | 9 | Intersection multiple |
             """)
         st.markdown("""
         **Analyse :** 
         - Les intersections **simples (X, T, Y)** concentrent le plus grand nombre d'accidents
         - Les **giratoires** génèrent beaucoup d'accidents mais **peu graves** (grâce à des vitesses modérées)
         - Les **échangeurs, passages à niveau ou intersections multiples** sont moins fréquents,
           mais leurs accidents sont souvent **plus violents**
         """)
         st.divider()

    # Distribution lum/gravite
    with col:
         name="Dist_lum_grav.png"
         st.image(f"reports/{name}", width=400, use_container_width=True)
         with st.expander("Définition des valeurs `lum`"):
            st.markdown("""
             | Code | Type d'intersection |
             |------|----------------------|
             | 1 | Plein jour |
             | 2 | Aube / crépuscule |
             | 3 | Nuit sans éclairage |
             | 4 | Nuit avec éclairage |
             | 5 | Nuit sans éclairage allumé |
            """)
         st.markdown("""
         **Analyse :** 
         - La **gravité augmente** lorsque la luminosité diminue
         - Les accidents survenant **la nuit sans éclairage** sont proportionnellement **les plus graves**
         - En **plein jour**, les accidents sont plus nombreux mais souvent **moins sévères**
         - La **visibilité** est donc un facteur déterminant dans la gravité des accidents
         """)
         st.divider()
     
    # Distribution secu/gravite
    with col:
         name="Dist_secu_grav.png"
         st.image(f"reports/{name}", width=400, use_container_width=True)
         with st.expander("Définition des valeurs `secu`"):
             st.markdown("""
              | Code | Signification |
              |------|----------------|
              | -1 | Donnée manquante / non renseignée |
              | 0 | Aucun dispositif de sécurité |
              | 1 | Ceinture de sécurité |
              | 2 | Casque |
              | 3 | Dispositif enfant |
              | 4 | Gilet réfléchissant |
              | 9 | Autre dispositif |
              """)
         st.markdown("""
         **Analyse :** 
         - L'absence de dispositif de sécurité (`secu = 0`) est clairement associée à une **gravité plus forte**
         - Le **port de la ceinture ou du casque** réduit fortement la proportion de tués
         - Ces résultats confirment l'impact majeur des **équipements de sécurité** dans la prévention des décès routiers
         """)
         st.divider()

    # Corr var expl/gravite
    with col:
         name="Corr_var_expl_Grav.png"
         st.image(f"reports/{name}", width=400, use_container_width=True)
         st.markdown("""
         **Analyse :** 
         - Aucune variable avec corrélation forte (> 0.3), mais certaines tendances directionnelles
         - La gravité dépend d'interactions complexes
         """)
         st.divider()

    # Corr var expl
    with col:
         name="Corr_var_expl.png"
         st.image(f"reports/{name}", width=400, use_container_width=True)
         st.markdown("""
         **Analyse :** 
         - Aucune corrélation forte (> 0.8) entre variables explicatives n'est observée
         - Cela confirme l'absence de **multicolinéarité majeure** dans le jeu de données
         - Chaque variable peut donc apporter une **information complémentaire** utile à la modélisation
         """)
         st.divider()



