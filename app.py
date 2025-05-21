import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

# Configuration de la page
st.set_page_config(page_title="Prévision EGT Margin – Moteur 802290", layout="centered")
st.title("Prévision EGT Margin – Moteur 802290")

# Affichage du logo si présent
def show_logo():
    if os.path.exists("logo.png"):
        st.image("logo.png", width=180)
show_logo()

# Chargement du fichier de prévision
file_path = "EGT_Margin_Forecast_802290_200_Cycles.xlsx"
if not os.path.exists(file_path):
    st.error(f"Fichier {file_path} introuvable dans le dossier.")
    st.stop()
df = pd.read_excel(file_path)

# --- PARTIE 1 : Validation du modèle ---
try:
    # Chargement des données d'entraînement
    df_train = pd.read_excel("802290 data ready to use.xlsx")
    y_true = df_train["EGT Margin"].dropna().values[-200:]  # Les 200 dernières vraies valeurs
    y_pred = df["EGT Margin Forecast (XGBoost)"].values[:len(y_true)]
    
    # Calcul des métriques
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Affichage des métriques dans un conteneur stylisé
    st.markdown("### 📊 Validation du modèle")
    st.markdown("""
    <style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R²", f"{r2:.3f}", help="Coefficient de détermination")
        col2.metric("MAE", f"{mae:.2f}°C", help="Erreur absolue moyenne")
        col3.metric("MSE", f"{mse:.2f}", help="Erreur quadratique moyenne")
        col4.metric("RMSE", f"{rmse:.2f}°C", help="Racine de l'erreur quadratique moyenne")
        st.markdown('</div>', unsafe_allow_html=True)
except Exception as e:
    st.warning("⚠️ Impossible d'afficher les métriques de validation (données réelles manquantes ou format inattendu).")

# --- PARTIE 2 : Graphique avec zone tolérancée ---
st.markdown("### 📈 Prévision de l'EGT Margin")
fig, ax = plt.subplots(figsize=(12, 6))

# Zone tolérancée verte
ax.axhspan(15, 18, color='red', alpha=0.2, label='Zone critique')

# Courbe de prévision
ax.plot(df['Date'], df['EGT Margin Forecast (XGBoost)'], 
        color="#2196F3", label="Prévision EGT Margin", linewidth=2)

# Personnalisation du graphique
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("EGT Margin (°C)", fontsize=12)
ax.set_title("Prévision EGT Margin – Moteur 802290", fontsize=15, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper right')

# Rotation des dates pour une meilleure lisibilité
fig.autofmt_xdate()

# Affichage du graphique
st.pyplot(fig)

# Bouton de téléchargement
with open(file_path, "rb") as f:
    st.download_button(
        label="📥 Télécharger le fichier Excel des prévisions",
        data=f,
        file_name="EGT_Margin_Forecast_802290_200_Cycles.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Style supplémentaire
st.markdown("""
<style>
    .stButton>button {
        background-color: #2196F3;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1.5em;
    }
    .stDownloadButton>button {
        background-color: #43A047;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1.5em;
    }
    .stMetric {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Configuration de la page Streamlit
st.set_page_config(page_title="Analyse EGT Margin", layout="wide")
st.title("Analyse de l'EGT Margin en fonction du CSN")

# Chargement du fichier Excel
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("802290 data ready to use.xlsx")
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {str(e)}")
        return None

# Chargement des données
df = load_data()

if df is not None:
    # Sélection des colonnes d'intérêt
    df = df[['EGT Margin', 'CSN']]
    
    # Nettoyage des données
    # Conversion des virgules en points si nécessaire
    df['EGT Margin'] = df['EGT Margin'].astype(str).str.replace(',', '.').astype(float)
    df['CSN'] = df['CSN'].astype(str).str.replace(',', '.').astype(float)
    
    # Suppression des lignes avec des valeurs NaN
    df = df.dropna()
    
    # Création du graphique avec Plotly
    fig = go.Figure()
    
    # Ajout de la ligne
    fig.add_trace(go.Scatter(
        x=df['CSN'],
        y=df['EGT Margin'],
        mode='lines+markers',
        name='EGT Margin',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))
    
    # Mise à jour du layout
    fig.update_layout(
        title="Évolution du EGT Margin en fonction du CSN",
        xaxis_title="CSN (Cycle Since New)",
        yaxis_title="EGT Margin (°C)",
        template="plotly_white",
        showlegend=True,
        hovermode="x unified"
    )
    
    # Ajout de la grille
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    # Affichage du graphique dans Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
    # Affichage des statistiques de base
    st.subheader("Statistiques descriptives")
    st.dataframe(df.describe())