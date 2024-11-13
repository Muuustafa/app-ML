import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Charger le dataset
df = pd.read_csv("Invistico_Airline.csv")

# Afficher le dataset en haut de l'application pour fournir une idée des données
st.title("Application de Prédiction de la Satisfaction Client")
st.write("Aperçu du Dataset :")
st.dataframe(df) 

# Identification des colonnes catégorielles
categorical_columns = df.select_dtypes(include=['object']).columns

# Encodage des variables catégorielles avec des variables factices
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

X = df.drop(columns=['satisfaction_satisfied'])
y = df['satisfaction_satisfied']

# Traiter les valeurs manquantes dans X
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Division du dataset en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instanciation des modèles
model_lr = LinearRegression()
model_knn = KNeighborsClassifier(n_neighbors=5)
model_nb = GaussianNB()

# Entraînement des modèles
model_lr.fit(X_train, y_train)
model_knn.fit(X_train, y_train)
model_nb.fit(X_train, y_train)

st.write("Entrez les valeurs pour tester les modèles :")

# Création de champs d'entrée pour les caractéristiques avec placeholders
input_data = {}
for col in X.columns:
    # Exemple de valeur moyenne pour le placeholder (ajustez selon vos besoins)
    example_value = X[col].mean() if pd.api.types.is_numeric_dtype(X[col]) else 0
    input_data[col] = st.number_input(
        f"Entrez la valeur pour {col}",
        value=float(example_value),
        format="%.4f"
    )

# Convertir les entrées utilisateur en DataFrame
input_df = pd.DataFrame([input_data])

# Aligner les colonnes d'entrée sur les colonnes du modèle (pour garantir la correspondance)
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Prédictions
if st.button("Prédire"):
    pred_lr = model_lr.predict(input_df)
    pred_knn = model_knn.predict(input_df)
    pred_nb = model_nb.predict(input_df)
    
    # Afficher les prédictions
    st.write(f"Prédiction (Régression Linéaire) : {pred_lr[0]}")
    st.write(f"Prédiction (KNN) : {pred_knn[0]}")
    st.write(f"Prédiction (Naïve Bayes) : {pred_nb[0]}")
