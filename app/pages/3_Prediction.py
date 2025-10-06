import streamlit as st
import pandas as pd
import joblib
import requests

st.title("🎯 Démo de prédiction")

# --- Charger le modèle et le sample ---

#source = st.radio("Jeu utilisé pour la démo :", ["ACCIDENT", "USAGER"], horizontal=True)

# ACCIDENT: Colonnes vraiment nécessaires
#FEATURES = ["lum","secu","col","moment","situ","catv","obs","grav_order"]  
#df = pd.read_csv("data/sample_merged_accident_mini.csv",
#                 usecols=FEATURES,
#                 nrows=5000,
#                 low_memory=False)


# USAGER: Colonnes vraiment nécessaires
#FEATURES = ["lum","secu","col","moment","situ","catv","obs","grav_order","is_we"]  
#
#df = pd.read_csv("data/sample_merged_usager_mini.csv",
#                 usecols=FEATURES,
#                 nrows=5000,
#                 low_memory=False)

FEATURES = ["lum","secu","col","obs", "catv","situ", "agg", "surf","atm"]  
df = pd.read_csv("data/X_test_encoded_sample.csv",
                 sep=",",
                 #skiprows=1,    # J'ai ajouté ce saut car une ligne est apparue dans ce df en 1ere ligne!??
                 nrows=100,
                 low_memory=False)
#df = df_full[(df_full[["lum","secu","col","obs", "catv","situ", "agg", "surf","atm"]] != -1).all(axis=1)]

y = pd.read_csv("data/y_test_encoded_sample.csv",
                sep=",",
                nrows=100)

base_url = "https://github.com/pestao233/DS_Accidents/releases/download/v1.0/"


st.write("Choisissez ou modifiez quelques variables ci-dessous pour tester le modèle :")

# Sélectionner une ligne de base
index = st.number_input("Numéro de ligne de référence", min_value=1, max_value=len(df)-1, value=1)
row = df.iloc[index].copy()

st.write("Voici la ligne choisie:")
st.dataframe(pd.DataFrame([row]), use_container_width=True)

st.info(f"👉 Gravité réelle (dans le dataset) : **{y.iloc[index,0]}**")

# Exemple : quelques variables à modifier
opts_lum = sorted(df["lum"].dropna().astype(int).unique().tolist())
lum = st.selectbox("Luminosité", opts_lum, index=opts_lum.index(int(row["lum"])) if int(row["lum"]) in opts_lum else 0)

opts_secu = sorted(df["secu"].dropna().astype(int).unique().tolist())
secu = st.selectbox("Moyen sécurité", opts_secu, index=opts_secu.index(int(row["secu"])) if int(row["secu"]) in opts_secu else 0)

opts_col = sorted(df["col"].dropna().astype(int).unique().tolist())
col = st.selectbox("Type de collision", opts_col, index=opts_col.index(int(row["col"])) if int(row["col"]) in opts_col else 0)

opts_obs = sorted(df["obs"].dropna().astype(int).unique().tolist())
obs = st.selectbox("Obstacle fixe heurté", opts_obs, index=opts_obs.index(int(row["obs"])) if int(row["obs"]) in opts_obs else 0)

opts_catv = sorted(df["catv"].dropna().astype(int).unique().tolist())
catv = st.selectbox("Catégorie du véhicule", opts_catv, index=opts_catv.index(int(row["catv"])) if int(row["catv"]) in opts_catv else 0)

opts_situ = sorted(df["situ"].dropna().astype(int).unique().tolist())
situ = st.selectbox("Situation de l'accident", opts_situ, index=opts_situ.index(int(row["situ"])) if int(row["situ"]) in opts_situ else 0)

opts_agg = sorted(df["agg"].dropna().astype(int).unique().tolist())
agg = st.selectbox("En agglomération", opts_agg, index=opts_agg.index(int(row["agg"])) if int(row["agg"]) in opts_agg else 0)

opts_surf = sorted(df["surf"].dropna().astype(int).unique().tolist())
surf = st.selectbox("Etat de la surface", opts_surf, index=opts_surf.index(int(row["surf"])) if int(row["surf"]) in opts_surf else 0)

opts_atm = sorted(df["atm"].dropna().astype(int).unique().tolist())
atm = st.selectbox("Conditions météo", opts_atm, index=opts_atm.index(int(row["atm"])) if int(row["atm"]) in opts_atm else 0)

# Mettre à jour la ligne avec les valeurs choisies
if "lum" in df.columns: # lum=3 (nuit sans éclairage)
    row["lum"] = lum
if "secu" in df.columns: #secu = 0 (aucun)
    row["secu"] = secu
if "col" in df.columns: # Col=1 (2 frontal), 5(collision multiple)
    row["col"] = col
#if "moment" in df.columns: #moment=3,4
#    row["moment"] = moment
if "obs" in df.columns: #obs=2 (arbre), 13 (fossé)
    row["obs"] = obs
if "catv" in df.columns: # catv=1(byciclette),2(cyclomoteur),30(scooter)-34
    row["catv"] = catv
if "situ" in df.columns: # situation accident=2 (bande arrêt urgence)
    row["situ"] = situ
if "agg" in df.columns: #agg=1 (hors agg)
    row["agg"] = agg
#if "is_we" in df.columns: #
#    row["is_we"] = is_we
if "surf" in df.columns: # Etat surface=7 verglacée
    row["surf"] = surf
if "atm" in df.columns: # atm=3(pluie forte), 5(brouillard), 6 (vent fort)
    row["atm"] = atm

st.write("Observation envoyée au modèle (après modification) :")
st.dataframe(pd.DataFrame([row]), use_container_width=True)



# Bouton de prédiction
if st.button("Lancer la prédiction"):
    X = pd.DataFrame([row])
    #X = X.drop(["grav_simpl"], axis=1)
    for file in ["Enora_scaler.joblib","Modele_Enora_rf.joblib"]:
        r = requests.get(base_url + file)
        open(file, "wb").write(r.content)

    scaler = joblib.load("Enora_scaler.joblib")
    rf = joblib.load("Modele_Enora_rf.joblib")

    X_scaled = scaler.transform(X)
    
    y_pred = rf.predict(X_scaled)[0]

    st.success(f"👉 Résultat de la prédiction : **{int(y_pred)}**")

    if hasattr(rf, "predict_proba"):
        st.caption(f"Confiance (proba max) : {rf.predict_proba(X).max():.3f}")
#caption("Le modèle utilise un pipeline pré-entraîné sur les données 2005–2018.")
