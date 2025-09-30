import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Accidents routiers - DataScientest - Cohorte DS JUL25",
    page_icon="🚧",
    layout="wide"
)

# Titre principal
st.title("🚧 Accidents routiers en France — Projet DataScientest / Equipe Enora Lever + Philippe Afonso")
st.caption("Démo Streamlit — exploration, modélisation, prédiction")

# Message d'accueil
st.markdown("""
Présentation du Projet sur **Streamlit** 

Utilisez le menu de gauche pour naviguer :
- **Exploration des données (variables explicatives + cible)** 
- **Préparation des données**
- **Modélisation** (entraînement de modèles RandomForest / HGB / .. + affichage de métriques)
- **Prédiction** : faire une prédiction sur de nouvelles données
""")

# Contexte
st.write("""<u>__Contexte__</u>
La sécurité routière constitue un enjeu majeur de santé publique et économique.
En France, plusieurs dizaines de milliers d’accidents corporels surviennent chaque année, provoquant des blessures, des décès et des coûts importants pour la société.
L’amélioration de la compréhension des facteurs influençant la gravité des accidents permettrait d’orienter les politiques publiques et  de mieux cibler les actions de sensibilisation et les campagnes de prévention.
Dans ce cadre, notre projet vise à analyser et à modéliser les données d’accidents routiers afin de prédire la gravité d’un accident à partir de variables explicatives liées au contexte, aux usagers et aux véhicules impliqués.
La problématique est un problème de classification supervisée : prédire une variable catégorielle (gravité) à partir de données hétérogènes (environnement, usagers, véhicules).
""")

# Objectifs
st.write("""__Objectifs__:
L’objectif principal de ce projet est de mettre en place un pipeline complet de Data Science comprenant :
- L’exploration et compréhension des données mises à disposition par le ministère de l’intérieur.
- L’identification des variables pertinentes et pré-traitement des données.
- La préparation d’un jeu de données exploitable pour une phase ultérieure de modélisation.
Notre démarche est donc progressive :
1.	partir d’une exploration descriptive des données,
2.	poursuivre avec un travail de visualisation et de pré-processing,
3.	avant d’aborder la modélisation proprement dite.""")
