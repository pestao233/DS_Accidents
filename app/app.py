import streamlit as st

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

# Configuration de la page
st.set_page_config(
    page_title="Accidents routiers - DataScientest - Cohorte DS JUL25",
    page_icon="🚧",
    layout="wide"
)

# Titre principal
st.title("🚧 Accidents routiers en France")
st.caption("Démo Streamlit — exploration, modélisation, prédiction")

with st.sidebar:
    st.markdown("### 👥 Équipe projet - Cohorte DS JUL25")
    st.markdown("""
    <div style='border:1px solid #ccc; border-radius:10px; padding:10px; background-color:#111827; color:#E5E7EB'>
    <b>Enora Lever</b><br>
    <b>Philippe Afonso</b>
    </div>
    """, unsafe_allow_html=True)
    
# Message d'accueil
st.markdown("""
Présentation du Projet sur **Streamlit** 

Utilisez le menu de gauche pour naviguer :
- **Exploration des données (variables explicatives + cible)** 
- **Modélisation** (entraînement de modèles + affichage de métriques)
- **Prédiction** : faire une prédiction sur de nouvelles données
""")

st.divider()

# Contexte
st.markdown('<h2>Contexte</h2>', unsafe_allow_html=True)
st.write("""La sécurité routière constitue un enjeu majeur de santé publique et économique.\n
En France, plusieurs dizaines de milliers d’accidents corporels surviennent chaque année, provoquant des blessures, des décès et des coûts importants pour la société.\n
L’amélioration de la compréhension des facteurs influençant la gravité des accidents permettrait d’orienter les politiques publiques et  de mieux cibler les actions de sensibilisation et les campagnes de prévention.\n
Dans ce cadre, notre projet vise à analyser et à modéliser les données d’accidents routiers afin de prédire la gravité d’un accident à partir de variables explicatives liées au contexte, aux usagers et aux véhicules impliqués.\n
La problématique est un problème de classification supervisée : prédire une variable catégorielle (gravité) à partir de données hétérogènes (environnement, usagers, véhicules).
""")

st.divider()


# Objectifs
st.markdown('<h2>Objectifs</h2>', unsafe_allow_html=True)
st.write("""
L’objectif principal de ce projet est de mettre en place un pipeline complet de Data Science comprenant :
- L’**exploration et compréhension** des données mises à disposition par le ministère de l’intérieur
- L’identification des **variables pertinentes** et pré-traitement des données
- La préparation d’un **jeu de données exploitable**
- La **modélisation** prédictive à l'aide d'algorithmes de machine learning
- L'**évaluation et la comparaison** des performances selon plusieurs métriques (precision, recall, F1-score)
- Et enfin, la mise en forme des livrables dans Streamlit (**visualiser** les analyses et de **tester** le modèle en temps réel)

Notre démarche est donc progressive :
1.	partir d’une **exploration** descriptive des données,
2.	poursuivre avec un travail de **visualisation et de pré-processing**,
3.	avant d’aborder la **modélisation** proprement dite.""")
