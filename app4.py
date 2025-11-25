import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------
# Carga de datos
# ---------------------------
@st.cache_data
def load_data(path: str = "equipment_anomaly_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

# ---------------------------
# Entrenamiento del modelo
# ---------------------------
@st.cache_resource
def train_model(df: pd.DataFrame):
    # Copia de seguridad
    df2 = df.copy()

    # Label encoders independientes para cada columna categórica
    le_equipment = LabelEncoder()
    le_location = LabelEncoder()

    df2["equipment"] = le_equipment.fit_transform(df2["equipment"])
    df2["location"] = le_location.fit_transform(df2["location"])

    # Separar X e y
    X = df2.drop("faulty", axis=1)
    y = df2["faulty"]

    # Columnas que NO se escalan
    columns_to_not_scale = ["location", "equipment"]

    # Separar columnas a escalar y no escalar
    X_to_scale = X.drop(columns=columns_to_not_scale)
    X_not_scaled = X[columns_to_not_scale]


    # Escalado
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X_to_scale)
    X_scaled_df_part = pd.DataFrame(
        X_scaled_array,
        columns=X_to_scale.columns,
        index=X.index
    )

    # Recombinar
    X_scaled_df = pd.concat([X_scaled_df_part, X_not_scaled], axis=1)

    # Recombinar
    X_scaled_df = pd.concat([X_scaled_df_part, X_not_scaled], axis=1)

    # Balanceo con SMOTE
    smote = SMOTE(sampling_strategy="auto", random_state=42)
    X_smote, y_smote = smote.fit_resample(X_scaled_df, y)

    # Split train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X_smote, y_smote, train_size=0.8, random_state=42, stratify=y_smote
    )

    # Modelo
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluación básica
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")


    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


    # Devolvemos todo lo necesario para usar el modelo luego
    return {
        "model": model,
        "scaler": scaler,
        "le_equipment": le_equipment,
        "le_location": le_location,
        "columns_to_not_scale": columns_to_not_scale,
        "feature_columns": X.columns.tolist(),
        "metrics": metrics,
        "df_original": df,
    }

# ---------------------------
# Preprocesado de una nueva muestra
# ---------------------------

def preprocess_single_sample(sample_df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """
    sample_df: DataFrame con columnas:
        temperature, pressure, vibration, humidity, location, equipment
        (location y equipment como strings originales)
    """
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    le_equipment = artifacts["le_equipment"]
    le_location = artifacts["le_location"]
    columns_to_not_scale = artifacts["columns_to_not_scale"]
    feature_columns = artifacts["feature_columns"]

    # Copia para no modificar el original
    df_user = sample_df.copy()

    # Codificar categóricas con los mismos encoders
    df_user["equipment"] = le_equipment.transform(df_user["equipment"])
    df_user["location"] = le_location.transform(df_user["location"])

    # Asegurar mismo orden de columnas que en el entrenamiento
    df_user = df_user[feature_columns]

    # Separar columnas a escalar y no escalar
    X_to_scale = df_user.drop(columns=columns_to_not_scale)
    X_not_scaled = df_user[columns_to_not_scale]

    # Escalado usando el scaler entrenado
    X_scaled_array = scaler.transform(X_to_scale)
    X_scaled_df_part = pd.DataFrame(
        X_scaled_array,
        columns=X_to_scale.columns,
        index=df_user.index
    )

    # Recombinar
    X_scaled_user = pd.concat([X_scaled_df_part, X_not_scaled], axis=1)

    return X_scaled_user


def saludar():
    st.write("¡Hola desde la función!")
    # st.write(f"Probabilidad estimada de fallo: **{prob_faulty:.2%}**")
# ---------------------------
# Interfaz de Streamlit
# ---------------------------

def main():


    st.set_page_config(
        page_title="Detección de Fallos en Equipos",
        page_icon="🛠️",
        layout="centered",
    )



    st.title("Detección de anomalías en equipos")
    st.write(
        "Aplicación de clasificación para predecir si un equipo industrial fallará "
        "a partir de variables de proceso."
    )

    # 1. Cargar datos
    try:
        df = load_data("equipment_anomaly_data.csv")
    except FileNotFoundError:
        st.error(
            "No se encontró el archivo `equipment_anomaly_data.csv` en la carpeta actual. "
            "Colócalo junto al script de Streamlit y vuelve a ejecutar la app."
        )
        return

    # st.subheader("Vista rápida del dataset")
    # st.dataframe(df.head())

    # 2. Entrenar modelo y mostrar métricas
    # st.subheader("Entrenamiento del modelo")
    with st.spinner("Entrenando modelo (una sola vez, se cachea)..."):
        artifacts = train_model(df)

    # metrics = artifacts["metrics"]
    # col1, col2, col3, col4 = st.columns(4)
    # col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    # col2.metric("Precision", f"{metrics['precision']:.3f}")
    # col3.metric("Recall", f"{metrics['recall']:.3f}")
    # col4.metric("F1-score", f"{metrics['f1']:.3f}")

    st.markdown("---")

    # 3. Formulario de entrada de usuario
    st.subheader("Ingresar condiciones de operación")

    df_original = artifacts["df_original"]

    # Rango de valores numéricos en el dataset para guiar los sliders
    temp_min, temp_max = float(0), float(100)
    press_min, press_max = float(0), float(df_original["pressure"].max()+100)
    vib_min, vib_max = float(0), float(10)
    hum_min, hum_max = float(0), float(100)

    locations = sorted(df_original["location"].unique())
    equipments = sorted(df_original["equipment"].unique())




    if st.button("Haz clic aquí"):
        saludar()




    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            temperature = st.number_input(
                "Temperatura [°C]",
                value=float(np.mean([temp_min, temp_max])),
                min_value=temp_min,
                max_value=temp_max,
            )
            vibration = st.number_input(
                "Vibración",
                value=float(np.mean([vib_min, vib_max])),
                min_value=vib_min,
                max_value=vib_max,
            )
            location = st.selectbox("Localización", options=locations)
        with c2:
            pressure = st.number_input(
                "Presión [bar]",
                value=float(np.mean([press_min, press_max])),
                min_value=press_min,
                max_value=press_max,
            )
            humidity = st.number_input(
                "Humedad [%]",
                value=float(np.mean([hum_min, hum_max])),
                min_value=hum_min,
                max_value=hum_max,
            )
            equipment = st.selectbox("Equipo", options=equipments)

        submitted = st.form_submit_button("Predecir fallo")

    # 4. Predicción
    if submitted:
        # Construir DataFrame de una fila con los datos de usuario
        sample_df = pd.DataFrame(
            {
                "temperature": [temperature],
                "pressure": [pressure],
                "vibration": [vibration],
                "humidity": [humidity],
                "location": [location],
                "equipment": [equipment],
            }
        )

        try:
            X_user_processed = preprocess_single_sample(sample_df, artifacts)
            model = artifacts["model"]

            y_pred = model.predict(X_user_processed)[0]
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_user_processed)[0]
                prob_faulty = proba[1] if len(proba) > 1 else None
            else:
                prob_faulty = None

            st.markdown("### Resultado de la predicción")

            if y_pred == 1:
                st.error("⚠️ El modelo predice que el equipo **fallará**")
            else:
                st.success("✅ El modelo predice que el equipo **NO** fallará.")

            if prob_faulty is not None:
                st.write(f"Probabilidad estimada de fallo: **{prob_faulty:.2%}**")

            with st.expander("Ver datos usados para esta predicción"):
                st.write(sample_df)

        except Exception as e:
            st.error(f"Ocurrió un error al generar la predicción: {e}")


if __name__ == "__main__":

    main()
