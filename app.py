import streamlit as st
import pandas as pd
import joblib
from unidecode import unidecode
from fuzzywuzzy import process

# ==============================
# 📥 Cargar modelo y datos
# ==============================
modelo = joblib.load("modelo_entrenado.pkl")
df = pd.read_csv("data_ciudades.csv", encoding='latin1', sep=";", engine='python', on_bad_lines='skip')
df.columns = df.columns.str.strip()

# ✅ Renombrar columnas para que coincidan con el modelo
df = df.rename(columns={
    '%_pm': '% pm',
    'direccion': 'dirección',
    'hechos_violentos': 'hechos violentos'
})

# ==============================
# 🧾 Mostrar columnas en el sidebar
# ==============================
st.sidebar.markdown("🧾 Columnas encontradas en el archivo:")
st.sidebar.write(df.columns.tolist())

# ==============================
# 📦 Interfaz principal
# ==============================
st.title("📦 Recomendador de Método de Entrega")
st.markdown("Descubre si deberías enviar **contraentrega** o con **pago anticipado**, según la ciudad destino.")

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

        # ✅ Calcular tasa de devolución
        entregas = float(fila['entregas']) if fila['entregas'] != 0 else 1  # Evitar división por cero
        devoluciones = float(fila['devoluciones'])
        tasa_dev = devoluciones / entregas

        # ✅ Crear input del modelo en orden exacto y con mismos nombres
        input_modelo = pd.DataFrame([{
            '% pm': float(fila['% pm']),
            'oficina': int(fila['oficina']),
            'dirección': int(fila['dirección']),
            'hechos violentos': int(fila['hechos violentos']),
            'tasa_devolucion': tasa_dev
        }])

        # ✅ Aseguramos el mismo orden que en el entrenamiento
        input_modelo = input_modelo[['% pm', 'oficina', 'dirección', 'hechos violentos', 'tasa_devolucion']]

        # ✅ Predicción
        try:
            pred = modelo.predict(input_modelo)[0]
            st.write(f"📈 Predicción del modelo: `{pred:.4f}`")

            if pred >= 0.5:
                st.success("✅ Puedes hacer la entrega **CONTRAENTREGA** con alta probabilidad de éxito.")
            else:
                st.error("⚠️ Se recomienda **PAGO ANTICIPADO** para evitar riesgo de devolución.")
        except Exception as e:
            st.error("🚨 Error al hacer la predicción. Revisa el formato de entrada.")
            st.text(str(e))
