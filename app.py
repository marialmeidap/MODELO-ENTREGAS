import streamlit as st
import pandas as pd
import joblib
from unidecode import unidecode
from fuzzywuzzy import process

# ========================
# Cargar modelo y datos
# ========================
modelo = joblib.load("modelo_entrenado.pkl")

# Leer CSV con manejo de errores y codificación
df = pd.read_csv("data_ciudades.csv", encoding='latin1', sep=",", engine='python', on_bad_lines='skip')

# Normalizar nombres de columnas para evitar errores
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("á", "a")
    .str.replace("é", "e")
    .str.replace("í", "i")
    .str.replace("ó", "o")
    .str.replace("ú", "u")
)

# Mostrar nombres de columnas (debug)
st.write("🧾 Columnas disponibles:", df.columns.tolist())

# ========================
# Interfaz
# ========================
st.title("📦 Recomendador de Método de Entrega")
st.markdown("Descubre si deberías enviar **contraentrega** o con **pago anticipado**, según la ciudad destino.")

# ========================
# Input del usuario
# ========================
ciudad_usuario = st.text_input("🔎 Ingresa el nombre de la ciudad destino:")

if ciudad_usuario:
    ciudad_proc = unidecode(ciudad_usuario.strip().lower())
    ciudades_df = df['ciudad'].dropna().astype(str).str.lower().apply(unidecode).unique()

    mejor_coincidencia, score = process.extractOne(ciudad_proc, ciudades_df)

    if score < 70:
        st.warning(f"⚠️ Ciudad '{ciudad_usuario}' no encontrada. Verifica la ortografía.")
    else:
        st.success(f"🔁 Ciudad más parecida encontrada: {mejor_coincidencia.upper()} (similitud: {score}%)")

        fila = df[df['ciudad'].str.lower().apply(unidecode) == mejor_coincidencia].iloc[0]

        # Crear el input exacto que espera el modelo
        input_modelo = pd.DataFrame([{
            'oficina': fila['oficina'],
            'direccion': fila['direccion'],  # Reemplazo si ya fue limpiado
            'hechos_violentos': fila['hechos_violentos'],
            'pm': fila['pm'],  # columna '% pm' normalizada a 'pm'
            'tasa_devolucion': fila['tasa_devolucion']
        }])

        # Si las columnas originales se llaman distinto, ajusta aquí:
        # st.write(input_modelo.columns)

        # ✅ Predicción
        pred = modelo.predict(input_modelo)[0]
        st.write(f"📈 Predicción del modelo: `{pred:.4f}`")

        if pred >= 0.5:
            st.success("✅ Puedes hacer la entrega **CONTRAENTREGA** con alta probabilidad de éxito.")
        else:
            st.error("⚠️ Se recomienda **PAGO ANTICIPADO** para evitar riesgo de devolución.")
