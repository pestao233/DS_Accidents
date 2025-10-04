import streamlit as st
import pandas as pd
import joblib

st.title("🎯 Démo de prédiction")

# --- Charger le modèle et le sample ---


# ACCIDENT: Colonnes vraiment nécessaires
FEATURES = ["lum","secu","col","moment","situ","catv","obs","grav_order"]  
df = pd.read_csv("data/sample_merged_accident_mini.csv",
                 usecols=FEATURES,
                 nrows=5000,
                 low_memory=False)


# USAGER: Colonnes vraiment nécessaires
FEATURES = ["lum","secu","col","moment","situ","catv","obs","grav_order"]  

df = pd.read_csv("data/sample_merged_usager_mini.csv",
                 usecols=FEATURES,
                 nrows=5000,
                 low_memory=False)


model = joblib.load("models/model_HGBC.joblib")

st.write("Choisissez ou modifiez quelques variables ci-dessous pour tester le modèle :")

# Sélectionner une ligne de base
index = st.number_input("Numéro de ligne de référence", min_value=0, max_value=len(df)-1, value=0)
row = df.iloc[index].copy()

st.write("Voici la ligne choisie:")
st.dataframe(row)

# Exemple : quelques variables à modifier (adapte les noms à ton jeu)
lum = st.selectbox("Luminosité", sorted(df["lum"].unique()), index=int(row["lum"]) if "lum" in df.columns else 0)
#atm = st.selectbox("Conditions météo", sorted(df["atm"].unique()), index=int(row["atm"]) if "atm" in df.columns else 0)
secu = st.selectbox("Moyen sécurité", sorted(df["secu"].unique()), index=int(row["secu"]) if "secu" in df.columns else 0)
col = st.selectbox("Type de collision", sorted(df["col"].unique()), index=int(row["col"]) if "col" in df.columns else 0)
#vitesse = st.slider("Vitesse du véhicule (km/h)", 0, 200, int(row["vitesse"]) if "vitesse" in df.columns else 50)

# Mettre à jour la ligne avec les valeurs choisies
if "lum" in df.columns:
    row["lum"] = lum
if "secu" in df.columns:
    row["secu"] = secu
if "col" in df.columns:
    row["col"] = col
#if "vitesse" in df.columns:
#    row["vitesse"] = vitesse

# Bouton de prédiction
if st.button("Lancer la prédiction"):
    X = pd.DataFrame([row])
    y_pred = model.predict(X)[0]
    st.success(f"👉 Résultat de la prédiction : **{y_pred}**")

st.caption("Le modèle utilise un pipeline pré-entraîné sur les données 2005–2018.")
