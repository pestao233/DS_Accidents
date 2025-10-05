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
tab1, tab2, tab3, tab4 = st.tabs(["📥 Chargement", "🔍 Exploration", "🧼 Nettoyage", "📊 Dataviz"])
with tab1:
    st.markdown("#### Aperçu (Caractéristiques)")
    st.dataframe(caracs.head())
    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes totales (full):", "958 469")
    c2.metric("Lignes totales (sample):", f"{caracs.shape[0]:,}".replace(",", " "))
    c3.metric("Colonnes totales:", caracs.shape[1])
    #c3.metric("Mémoire (Mo)", round(caracs.memory_usage(deep=True).sum()/1024**2, 2))

    st.markdown("#### Aperçu (Usagers)")
    st.dataframe(usagers.head())
    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes totales (full):", "2 142 195")
    c2.metric("Lignes totales (sample):", f"{usagers.shape[0]:,}".replace(",", " "))
    c3.metric("Colonnes totales:", usagers.shape[1])
    #c3.metric("Mémoire (Mo)", round(usagers.memory_usage(deep=True).sum()/1024**2, 2))
   
    st.markdown("#### Aperçu (Lieux)")
    st.dataframe(lieux.head())
    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes totales (full):", "958 469")
    c2.metric("Lignes totales (sample):", f"{lieux.shape[0]:,}".replace(",", " "))
    c3.metric("Colonnes totales:", lieux.shape[1])
    #c3.metric("Mémoire (Mo)", round(lieux.memory_usage(deep=True).sum()/1024**2, 2))
            
    st.markdown("#### Aperçu (Véhicules)")
    st.dataframe(vehicules.head())
    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes totales (full):", "1 635 811")
    c2.metric("Lignes totales (sample):", f"{vehicules.shape[0]:,}".replace(",", " "))
    c3.metric("Colonnes totales:", vehicules.shape[1])
    #c3.metric("Mémoire (Mo)", round(vehicules.memory_usage(deep=True).sum()/1024**2, 2))  

with tab2:
    # info()
    # Carac
    st.subheader("Résumé du DataFrame `Caractéristiques` (df.info)")
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


    # Usagers
    st.subheader("Résumé du DataFrame `Usagers` (df.info)")
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
    
    # Lieux
    st.subheader("Résumé du DataFrame `Lieux` (df.info)")
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
            
    # Vehicules
    st.subheader("Résumé du DataFrame `Vehicules` (df.info)")
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

            
with tab3:
    st.markdown("#### Nettoyages appliqués")
    # ⬇️ colle ici tes transformations (rename, types, fillna...), puis un aperçu

with tab4:
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
         - Baisse nette et continue entre 2005 et 2018 - coïncide avec l'instalation de plus de 500 radars en 2005
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
         - L'après-midi est la période la plus accidentogène (38.5%) 45 corrélé à un trafic élevé et à un rythme de circulation actif  
         - Le soir suit en intensité (baisse de luminosité + fatigue accrue)
         - Le matin a une contribution modérée  
         - La nuit (0h-6h) est la moins fréquentée, mais les accidents qui y surviennent peuvent être plus graves  
         - À renforcer : l'analyse de la **gravité** par plage horaire, pour distinguer "nombre" vs "impact"
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



