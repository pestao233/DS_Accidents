import streamlit as st
import pandas as pd
from src.modeling import load_model
import os

st.title("Prédiction")
st.write("Prédire avec de nouvelles données")

