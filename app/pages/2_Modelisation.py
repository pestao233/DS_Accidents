import streamlit as st
import pandas as pd
import os

st.title("Modélisation")
st.write("""Nous avons encodé la variable gravité afin d’obtenir une seule ligne par accident, correspondant à la gravité maximale observée.
La classe Indemne a ainsi disparu, laissant un modèle de prédiction sur trois niveaux de gravité : Blessé léger, Blessé hospitalisé et Tué.""")

st.write("#### Nous avons testé différents modèles ")

st.write("###### RandomForest ")
with st.expander("Voir le résultat"):
  st.image("reports/RandomForest.png", use_container_width=True)

st.write("###### LogisticRegression ")
with st.expander("Voir le résultat"):
  st.image("reports/LogisticRegression.png", use_container_width=True)

st.write("###### HistGradientBoosting ")
with st.expander("Voir le résultat"):
  st.image("reports/HistGradientBoosting.png", use_container_width=True)

st.write("###### BalancedRandomForestClassifier")
with st.expander("Voir le résultat"):
  st.image("reports/BalancedRandomForest.png", use_container_width=True)

st.write("###### EasyEnsembleClassifier")
with st.expander("Voir le résultat"):
  st.image("reports/EasyEnsembleClassifier.png", use_container_width=True)

## A modifier
st.write("""Nous avons selectionné les modèles les plus performants : BalancedRandomForest, RandomForest et HistGradientBoosting. 
Ils présentent les meilleurs scores F1_macro et resistent mieux au déséquilibre entre les classes""")

st.write("#### On cherche les meilleurs paramètres ")
st.write(""" Nous avons pris les meilleurs modèles et testés les hyperparamètres.
         
Etant toujours confrontés à des problèmes de mémoire, nous avons minimisé les paramètres à tester, de manière à éviter les bugs""")

st.write("###### HistGradientBoosting ")
st.code("""
# Modèle de base
hgb = HistGradientBoostingClassifier(random_state=42)

# Grille d’hyperparamètres
param_dist = {
    "learning_rate": uniform(0.05, 0.1),
    "max_iter": randint(100, 200),
    "max_depth": randint(3, 6),
    "min_samples_leaf": randint(20, 60)
}

# RandomizedSearch
search = RandomizedSearchCV(
    hgb,
    param_distributions=param_dist,
    n_iter=5,        
    cv=2,
    scoring="f1_macro",
    n_jobs=1,
    verbose=2,
    random_state=42
)

search.fit(X_train_bal, y_train_bal)

# Évaluation
best_hgb = search.best_estimator_
y_pred = best_hgb.predict(X_test_scaled)

print("✅ Best params:", search.best_params_)
print("✅Best CV F1_macro:", search.best_score_)
print("✅ Test set report :")
print(classification_report(y_test, y_pred))
        """, language="python")

with st.expander("Voir le résultat"):
  st.image("reports/HistGradientBoosting_Hyperparam.png", use_container_width=True)

st.write("###### RandomForest ")
with st.expander("Voir le résultat"):
  st.image("reports/RandomForest_Hyperparam.png", use_container_width=True)

# ajouter
st.write("###### BalancedRandomForest ")
with st.expander("Voir le résultat"):
  st.image("reports/Balanced_Random_hyperparam_debut.png", use_container_width=True)
  st.image("reports/Balanced_Random_hyperparam.png", use_container_width=True)











