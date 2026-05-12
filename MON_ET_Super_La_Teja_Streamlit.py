#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ==========================================================
# APP STREAMLIT – GASOLINA SUPER (MON_ET) – MODO PRO
# Gradient Boosting + Control de Confiabilidad Metrológica
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import warnings

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIGURACIÓN UI
# ==========================================================

st.set_page_config(
    page_title="MON_ET – Gasolina Super La Teja",
    page_icon="🧪",
    layout="centered"
)

# UI limpia (sacar link y padding)
st.markdown("""
<style>
a[href^="#"] { display: none !important; }
.block-container { padding-top: 2rem; }
.big-font { font-size:22px !important; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🧪 Estimación de MON_ET - Super La Teja")

# ==========================================================
# CRITERIOS METROLÓGICOS (tus valores)
# ==========================================================

REPRO_METODO = 0.83
UMBRAL_METODO = REPRO_METODO / 2      # 0.415 → ~0.42 como en tu comentario

SIGMA_ANALITICO = 0.42
n_sim = 100

# ==========================================================
# CARGA MODELO
# ==========================================================

@st.cache_resource
def cargar_modelo():
    modelo = load("Modelo_MON_Super.joblib")
    columnas = load("Columnas_MON_Super.joblib")
    return modelo, columnas

try:
    GBR, columnas_modelo = cargar_modelo()
    st.success("✅ Modelo Gradient Boosting con validación metrológica")
except Exception as e:
    st.error("❌ Error al cargar el modelo o las columnas (.joblib)")
    st.caption(f"Detalle técnico: {e}")
    st.stop()

# ==========================================================
# INPUT
# ==========================================================

archivo = st.file_uploader("📁 Cargar archivo CSV del LIMS", type=["csv"])

# ==========================================================
# FUNCIONES
# ==========================================================

def extraer_valor(df, nombre):
    """
    Busca en columna 1 el 'nombre' y devuelve columna 4.
    Si no está, retorna NaN.
    """
    fila = df[df[1] == nombre]
    if fila.empty:
        return np.nan
    return fila.iloc[0, 4]

def convertir_a_float(v):
    """
    Conversión robusta: '12,3' -> 12.3; si falla -> NaN
    """
    if pd.isna(v):
        return np.nan
    try:
        return float(str(v).replace(",", "."))
    except:
        return np.nan

def armar_df_pred(muestra, columnas_modelo):
    """
    Arma el DataFrame con las variables del modelo en el orden correcto.
    """
    datos = {
        'Densidad a 15º': extraer_valor(muestra, "Densidad promedio a 15º"),
        'Punto Inicial':  extraer_valor(muestra, "Punto Inicial"),
        '10% vol':        extraer_valor(muestra, "10% vol"),
        '20% vol':        extraer_valor(muestra, "20% vol"),
        '30% vol':        extraer_valor(muestra, "30% vol"),
        '40% vol':        extraer_valor(muestra, "40% vol"),
        '50% vol':        extraer_valor(muestra, "50% vol"),
        '60% vol':        extraer_valor(muestra, "60% vol"),
        '70% vol':        extraer_valor(muestra, "70% vol"),
        '80% vol':        extraer_valor(muestra, "80% vol"),
        '90% vol':        extraer_valor(muestra, "90% vol"),
        '95% vol':        extraer_valor(muestra, "95% vol"),
        'Punto Final':    extraer_valor(muestra, "Punto Final"),
        'T_VAP_ET':       extraer_valor(muestra, "Tensión de Vapor (con Etanol)")
    }

    # Convertir valores
    datos_convertidos = {k: convertir_a_float(v) for k, v in datos.items()}

    # DF en el orden del modelo
    df_pred = pd.DataFrame([datos_convertidos])

    # Reordenar/seleccionar columnas del modelo (evita error si el dict trae extras)
    df_pred = df_pred.reindex(columns=columnas_modelo)

    return df_pred, datos_convertidos

def monte_carlo_std(modelo, df_base, n_sim, sigma, factor):
    """
    Simula variando inputs con ruido normal y retorna std de predicciones.
    """
    preds = []
    for _ in range(n_sim):
        df_sim = df_base.copy()
        for col in df_sim.columns:
            ruido = np.random.normal(0, sigma * factor)
            df_sim[col] = df_sim[col] + ruido
        preds.append(modelo.predict(df_sim)[0])
    preds = np.asarray(preds)
    return preds.std()

# ==========================================================
# BOTÓN PRINCIPAL
# ==========================================================

if archivo is not None:

    if st.button("🚀 Calcular MON_ET"):

        with st.spinner("Procesando muestra..."):

            # Leer CSV (igual que tu script)
            try:
                muestra = pd.read_csv(archivo, sep=";", encoding="latin1", header=None)
            except Exception as e:
                st.error("❌ No se pudo leer el CSV (separador/encoding/formato)")
                st.caption(f"Detalle técnico: {e}")
                st.stop()

            # Campos clave
            try:
                celda_producto = muestra.loc[muestra[0] == "Producto", 4].values[0]
                celda_lims = muestra.loc[muestra[0] == "Número de Muestra", 4].values[0]
                celda_sp = muestra.loc[muestra[0] == "SamplingPoint", 4].values[0]
            except Exception:
                st.error("❌ Formato de archivo inválido (faltan campos: Producto / Número de Muestra / SamplingPoint)")
                st.stop()

            # Validación sampling point
            if celda_sp != "R-TK_FINAL_TEJA":
                st.error("❌ La muestra NO corresponde a Gasolina de TK FINAL TEJA.")
                st.warning(f"SamplingPoint encontrado: {celda_sp}")
                st.stop()

            # Validación producto
            if celda_producto != "GAS_SUP_95":
                st.error("❌ La muestra NO corresponde a GASOLINA SUPER (GAS_SUP_95).")
                st.warning(f"Producto encontrado: {celda_producto}")
                st.stop()

            # Armar DF para predicción
            df_pred, datos_convertidos = armar_df_pred(muestra, columnas_modelo)

            # Chequeo de faltantes (muy recomendable en producción)
            faltantes = [k for k, v in datos_convertidos.items() if (isinstance(v, float) and np.isnan(v))]
            if faltantes:
                st.error("❌ Datos incompletos. No se puede estimar MON_ET con confiabilidad.")
                st.warning("Faltan ensayos / variables:")
                st.write(", ".join(faltantes))
                st.stop()

            # Predicción principal
            try:
                mon_et_estimado = float(GBR.predict(df_pred)[0])
                mon_et_estimado = np.round(mon_et_estimado, 1)
            except Exception as e:
                st.error("❌ Error al predecir (revisar columnas/valores).")
                st.caption(f"Detalle técnico: {e}")
                st.stop()

            # Monte Carlo para error (std)
            mon_et_std = monte_carlo_std(
                modelo=GBR,
                df_base=df_pred,
                n_sim=n_sim,
                sigma=SIGMA_ANALITICO,
                factor=factor_ruido
            )
            error_reportado = np.round(mon_et_std, 2)

            # ======================================================
            # SEMÁFORO METROLÓGICO (tus reglas)
            # ======================================================
            if mon_et_std <= UMBRAL_METODO:
                color = "green"
                estado = "Resultado equivalente al método ASTM D 2700"
                icono = "🟢"
                mostrar_valor = True
            else:
                color = "red"
                estado = "Resultado NO confiable. Recomendar ASTM D 2700"
                icono = "🔴"
                mostrar_valor = True  # si querés ocultar el valor cuando no es confiable, poné False

            # ======================================================
            # RESULTADO VISUAL PRO
            # ======================================================
            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🔢 MON_ET estimado")

                if mostrar_valor:
                    valor = str(mon_et_estimado).replace(".", ",")
                    color_val = "black" if mon_et_std <= UMBRAL_METODO else "red"
                else:
                    valor = "❌"
                    color_val = "red"

                st.markdown(
                    f"""
                    <div style="
                        text-align: center;
                        font-size: 34px;
                        font-weight: bold;
                        color: {color_val};
                    ">
                        {valor}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.caption(f"Error (σ) estimado por Monte Carlo: **{str(error_reportado).replace('.', ',')}**")

            with col2:
                st.markdown(f"### 📋 LIMS: {celda_lims}")
                st.caption(f"Producto: **{celda_producto}**")
                st.caption(f"SamplingPoint: **{celda_sp}**")

            st.markdown("---")

            st.markdown(
                f"""
                <div style="text-align:center;">
                    <h2 style="color:{color};">{icono} {estado}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

