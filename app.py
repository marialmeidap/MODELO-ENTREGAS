import streamlit as st
import pandas as pd
import joblib
from unidecode import unidecode
from fuzzywuzzy import process

# ====================================
# 📥 Cargar modelo y datos
# ====================================
modelo = joblib.load("modelo_entrenado.pkl")
df = pd.read_csv("data_ciudades.csv", encoding='latin1', sep=";", engine='python', on_bad_lines='skip')
df.columns = df.columns.str.strip()

# ✅ Renombrar columnas para que coincidan con el modelo entrenado
df = df.rename(columns={
    '%_pm': '% pm',
    'direccion': 'dirección',
    'hechos_violentos': 'hechos violentos'
})

# ====================================
# 🧾 Diagnóstico de columnas
# ====================================
st.sidebar.markdown("🧾 Columnas encontradas en el archivo:")
st.sidebar.write(df.columns.tolist())

# ====================================
# 🧠 App principal
# ====================================
st.title("📦 Recomendador de Método de Entrega")
st.markdown("Descubre si deberías enviar **contraentrega** o con **pago anticipado**, según la ciudad destino.")

# 🔎 Input del usuario
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

        # ✅ Calcular tasa de devolución en tiempo real
        try:
            entregas = float(fila['entregas'])
            devoluciones = float(fila['devoluciones'])
            tasa_dev = devoluciones / entregas if entregas > 0 else 0
        except:
            tasa_dev = 0

        # ✅ Crear input para el modelo con columnas exactas
        input_modelo = pd.DataFrame([{
            '% pm': fila['% pm'],
            'oficina': fila['oficina'],
            'dirección': fila['dirección'],
            'hechos violentos': fila['hechos violentos'],
            'tasa_devolucion': tasa_dev
        }])

        # ✅ Reordenar columnas para que coincidan con el entrenamiento
        input_modelo = input_modelo[['% pm', 'oficina', 'dirección', 'hechos violentos', 'tasa_devolucion']]

        # ✅ Predicción
        pred = modelo.predict(input_modelo)[0]
        st.write(f"📈 Predicción del modelo: `{pred:.4f}`")

        # ✅ Interpretación
        if pred >= 0.5:
            st.success("✅ Puedes hacer la entrega **CONTRAENTREGA** con alta probabilidad de éxito.")
        else:
            st.error("⚠️ Se recomienda **PAGO ANTICIPADO** para evitar riesgo de devolución.")
