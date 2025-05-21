import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import datetime, timedelta

# Configuration des paramètres
LAG_SIZE = 50  # Nombre de cycles précédents à utiliser
FORECAST_HORIZON = 200  # Nombre de cycles à prédire

def load_and_prepare_data(file_path):
    """
    Charge et prépare les données depuis le fichier Excel
    """
    # Chargement des données
    df = pd.read_excel(file_path)
    
    # Conversion de la colonne de date
    df['Flight DateTime'] = pd.to_datetime(df['Flight DateTime'])
    
    # Sélection et conversion des colonnes nécessaires
    df = df[['Flight DateTime', 'EGT Margin', 'Vibration of the core', 'CSN']]
    
    # Conversion des colonnes numériques
    df['EGT Margin'] = pd.to_numeric(df['EGT Margin'], errors='coerce')
    df['Vibration of the core'] = pd.to_numeric(df['Vibration of the core'], errors='coerce')
    df['CSN'] = pd.to_numeric(df['CSN'], errors='coerce')
    
    # Suppression des lignes avec des valeurs manquantes
    df = df.dropna()
    
    return df

def create_lag_features(data, lag_size):
    """
    Crée les features de lag pour l'EGT Margin
    """
    lags = []
    for i in range(1, lag_size + 1):
        data[f'EGT_Margin_lag_{i}'] = data['EGT Margin'].shift(i)
        lags.append(f'EGT_Margin_lag_{i}')
    
    # Suppression des lignes avec des valeurs manquantes dues aux lags
    data = data.dropna()
    
    return data, lags

def prepare_training_data(df, lags):
    """
    Prépare les données pour l'entraînement du modèle
    """
    X = df[lags + ['Vibration of the core', 'CSN']]
    y = df['EGT Margin']
    
    return X, y

def generate_future_dates(last_date, n_periods):
    """
    Génère les dates futures pour les prévisions
    """
    # Supposons que chaque cycle représente environ 1 jour
    future_dates = [last_date + timedelta(days=i) for i in range(1, n_periods + 1)]
    return future_dates

def main():
    # Chargement et préparation des données
    print("Chargement des données...")
    df = load_and_prepare_data("802290 data ready to use.xlsx")
    
    # Création des features de lag
    print("Création des features de lag...")
    df, lags = create_lag_features(df, LAG_SIZE)
    
    # Préparation des données d'entraînement
    print("Préparation des données d'entraînement...")
    X, y = prepare_training_data(df, lags)
    
    # Entraînement du modèle
    print("Entraînement du modèle XGBoost...")
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X, y)
    
    # Préparation des données pour la prédiction
    last_lags = df['EGT Margin'].tail(LAG_SIZE).values.tolist()
    last_vibration = df['Vibration of the core'].iloc[-1]
    last_csn = df['CSN'].iloc[-1]
    last_date = df['Flight DateTime'].iloc[-1]
    
    # Génération des prévisions
    print("Génération des prévisions...")
    forecasts = []
    current_lags = last_lags.copy()
    
    for i in range(FORECAST_HORIZON):
        # Préparation des features pour la prédiction
        input_data = current_lags + [last_vibration, last_csn + i]
        pred = model.predict(np.array([input_data]))[0]
        forecasts.append(pred)
        
        # Mise à jour des lags pour la prochaine prédiction
        current_lags = [pred] + current_lags[:-1]
    
    # Création du DataFrame des résultats
    future_dates = generate_future_dates(last_date, FORECAST_HORIZON)
    results_df = pd.DataFrame({
        'Date': future_dates,
        'EGT Margin Forecast (XGBoost)': forecasts
    })
    
    # Export des résultats
    print("Export des résultats...")
    results_df.to_excel("EGT_Margin_Forecast_802290_200_Cycles.xlsx", index=False)
    print("Prévisions terminées et exportées avec succès!")

if __name__ == "__main__":
    main() 