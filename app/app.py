import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Accidents routiers - DS",
    page_icon="🚧",
    layout="wide"
)

# Titre principal
st.title("🚧 Accidents routiers en France — Projet DataScientest")
st.caption("Démo Streamlit — exploration, modélisation, prédiction")

# Message d'accueil
st.markdown("""
Bienvenue dans notre application **Streamlit** 🎉

Utilisez le menu de gauche pour naviguer :
- **Exploration** : charger un CSV d'exemple, explorer les variables et la cible
- **Modélisation** : entraîner un modèle baseline (RandomForest / HGB) et afficher les métriques
- **Prédiction** : faire une prédiction sur un fichier CSV
""")

# Petit message d'aide
st.info("Astuce : si aucun modèle n'est présent dans `models/`, allez dans la page *Modélisation* pour en créer un rapidement.")
