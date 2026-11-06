from pathlib import Path
from hashlib import sha256

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "svc_genetic_telco.pkl"
RESULTS_DIR = BASE_DIR / "resultados"

st.set_page_config(page_title="NexoRetain | Retención inteligente", page_icon="✦", layout="wide")
st.markdown("""
<style>
 .stApp {background: #0B1324; color: #E9EFFD;}
 [data-testid="stSidebar"] {background: #111D34; border-right: 1px solid #283E5E;}
 [data-testid="stMetric"] {background: #14233B; border: 1px solid #293F62;
   border-radius: 15px; padding: 18px;}
 [data-testid="stMetricLabel"] {color: #B8C8E3;}
 .brand-hero {background: linear-gradient(110deg, #233A64, #102A43 65%, #0E6970);
   padding: 30px; border-radius: 20px; border: 1px solid #40618C; margin-bottom: 18px;}
 .brand-hero h1 {font-size: 2.2rem; color: white; margin: 0 0 8px;}
 .brand-hero p {color: #D4E6FA; font-size: 1.05rem; margin: 0;}
 .brand-tag {color:#8FF1E2; letter-spacing:2px; font-size:.75rem; font-weight:750;}
 .brand-card {background:#14233B; border:1px solid #293F62; border-radius:15px;
   padding:20px; min-height:135px;}
 .brand-card h4 {color:#F1F7FF; margin:0 0 8px;}
 .brand-card p {color:#B8C8E3; margin:0;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def cargar_datos(path):
    data = pd.read_csv(path)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    return data


@st.cache_resource
def cargar_modelo(path, version_archivo):
    # La caché se renueva automáticamente cuando se reemplaza el modelo.
    return joblib.load(path)


def cargar_csv_opcional(nombre):
    path = RESULTS_DIR / nombre
    return pd.read_csv(path) if path.exists() else None


def obtener_modelo():
    if not MODEL_PATH.exists():
        st.error("Falta svc_genetic_telco.pkl. Expórtalo desde el notebook y cópialo aquí.")
        st.stop()
    try:
        modelo = cargar_modelo(MODEL_PATH, MODEL_PATH.stat().st_mtime_ns)
    except Exception as exc:
        st.error(f"No se pudo cargar el modelo: {exc}")
        st.stop()
    return modelo


def mostrar_matriz(matriz, titulo):
    fig = px.imshow(
        np.asarray(matriz), text_auto=True,
        x=["Pred. No Churn", "Pred. Churn"],
        y=["Real No Churn", "Real Churn"],
        color_continuous_scale="Blues", title=titulo,
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


def preparar_clientes(datos, columnas):
    """Validar esquema de entrada con los tipos usados durante el entrenamiento."""
    faltantes = [col for col in columnas if col not in datos.columns]
    if faltantes:
        raise ValueError("Faltan columnas obligatorias: " + ", ".join(faltantes))
    datos = normalizar_vacios(datos.loc[:, columnas])
    for col in ("tenure", "MonthlyCharges", "TotalCharges"):
        datos[col] = pd.to_numeric(datos[col], errors="coerce")
    # Evita que un CSV con valores ausentes convierta 0/1 en '0.0'/'1.0'.
    senior = pd.to_numeric(datos["SeniorCitizen"], errors="coerce")
    datos["SeniorCitizen"] = senior.map({0: "0", 1: "1"}).fillna("missing")
    for col in datos.columns.difference(["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]):
        datos[col] = datos[col].astype("string").fillna("missing").astype(str)
    return datos


def puntuar_cartera(datos, modelo, columnas):
    procesado = preparar_clientes(datos, columnas)
    score = modelo.predict_proba(procesado)[:, 1]
    clases = modelo.predict(procesado)
    resultado = datos.copy().reset_index(drop=True)
    resultado["P_Abandono"] = np.round(score, 4)
    resultado["Prediccion"] = np.where(clases == 1, "Churn", "No Churn")
    resultado["Prioridad"] = np.select(
        [score >= 0.7, score >= 0.4], ["Alta", "Media"], default="Baja")
    resultado["Accion sugerida"] = np.select(
        [score >= 0.7, score >= 0.4],
        ["Contactar y evaluar oferta", "Revisión comercial"],
        default="Seguimiento rutinario")
    return resultado


def descargar_csv(datos):
    return datos.to_csv(index=False).encode("utf-8-sig")


def normalizar_vacios(datos):
    """Convierte espacios o cadenas vacías en NaN para la imputación del pipeline."""
    limpio = datos.copy()
    for columna in limpio.select_dtypes(include=["object", "string"]).columns:
        limpio[columna] = limpio[columna].replace(r"^\s*$", np.nan, regex=True)
    return limpio


def validar_cartera(datos, columnas):
    """Devuelve incidencias que impedirían puntuar una cartera de forma fiable."""
    datos = normalizar_vacios(datos)
    errores = []
    if datos.empty:
        errores.append("El archivo no contiene clientes.")
    if len(datos) > 10000:
        errores.append("El límite de esta demostración es 10.000 clientes por archivo.")
    faltantes = [c for c in columnas if c not in datos.columns]
    if faltantes:
        errores.append("Faltan columnas obligatorias: " + ", ".join(faltantes))
    if datos.columns.duplicated().any():
        errores.append("El CSV contiene nombres de columnas repetidos.")
    if not faltantes:
        for col in ("tenure", "MonthlyCharges", "TotalCharges"):
            original = datos[col]
            numerico = pd.to_numeric(original, errors="coerce")
            invalidos = original.notna() & numerico.isna()
            if invalidos.any():
                errores.append(f"{col}: {invalidos.sum()} valores no numéricos.")
            if numerico.lt(0).any():
                errores.append(f"{col}: no se admiten valores negativos.")
        senior = pd.to_numeric(datos["SeniorCitizen"], errors="coerce")
        if (datos["SeniorCitizen"].notna() & ~senior.isin([0, 1])).any():
            errores.append("SeniorCitizen debe contener únicamente 0, 1 o valores vacíos.")
    return errores


def evaluar_campana(datos, capacidad, eficacia, valor, costo, criterio):
    """Simulación explícita, no estimación validada de retención incremental."""
    puntuados = datos.copy()
    puntuados["Valor esperado ($)"] = (
        puntuados["P_Abandono"] * eficacia * valor - costo
    ).round(2)
    columna = "P_Abandono" if criterio == "Mayor riesgo" else "Valor esperado ($)"
    puntuados = puntuados.sort_values([columna, "P_Abandono"], ascending=False, kind="stable")
    contactos = puntuados.head(capacidad).copy()
    return puntuados, contactos, float(contactos["Valor esperado ($)"].sum())


def agrupar_contribuciones_shap(nombres, valores, originales):
    """Suma columnas derivadas (One-Hot, escalado) en su variable de origen."""
    suma = {nombre: 0.0 for nombre in originales}
    for nombre_transformado, valor in zip(nombres, valores):
        sin_prefijo = nombre_transformado.split("__", 1)[-1]
        coincidencias = [c for c in originales if sin_prefijo == c or sin_prefijo.startswith(c + "_")]
        if coincidencias:
            original = max(coincidencias, key=len)
            suma[original] += float(valor)
    return pd.DataFrame({"Característica": list(suma), "Contribución SHAP": list(suma.values())})


def resumen_matriz_confusion(matriz):
    """Lee una matriz [[VN, FP], [FN, VP]] sin confundir clases."""
    m = np.asarray(matriz, dtype=int)
    if m.shape != (2, 2) or (m < 0).any():
        raise ValueError("Se requiere una matriz 2x2 con conteos no negativos.")
    vn, fp, fn, vp = m.ravel()
    return {"VN": int(vn), "FP": int(fp), "FN": int(fn), "VP": int(vp),
            "Recall": vp / (vp + fn) if vp + fn else 0.0,
            "Precision": vp / (vp + fp) if vp + fp else 0.0}


def simular_diferencia_comercial(matriz_base, matriz_nueva, efectividad, valor_retenido,
                                costo_contacto, tarifa):
    """Diferencia de campañas al contactar todos los positivos predichos."""
    base = resumen_matriz_confusion(matriz_base)
    nuevo = resumen_matriz_confusion(matriz_nueva)
    tp_extra = nuevo["VP"] - base["VP"]
    fp_extra = nuevo["FP"] - base["FP"]
    contactos_extra = (nuevo["VP"] + nuevo["FP"]) - (base["VP"] + base["FP"])
    retenidos_extra = tp_extra * efectividad
    beneficio = retenidos_extra * valor_retenido
    gasto = contactos_extra * costo_contacto + tarifa
    return {"tp_extra": tp_extra, "fp_extra": fp_extra,
            "contactos_extra": contactos_extra, "retenidos_extra": retenidos_extra,
            "ingreso_supuesto": beneficio, "costo_supuesto": gasto,
            "balance": beneficio - gasto}


st.markdown("""<div class="brand-hero"><div class="brand-tag">✦ NEXORETAIN · CUSTOMER INTELLIGENCE</div>
<h1>Clientes que entiendes. Relaciones que conservas.</h1>
<p>Analítica de abandono, priorización comercial y decisiones explicables con SVC genético.</p></div>""",
unsafe_allow_html=True)

if not DATA_PATH.exists():
    st.error(f"No se encuentra el CSV en: {DATA_PATH}")
    st.stop()

df = cargar_datos(DATA_PATH)
columnas_entrada = [c for c in df.columns if c not in ("customerID", "Churn")]

pagina = st.sidebar.radio(
    "✦ NexoRetain | Navegación",
    ["Inicio", "EDA", "Centro de retención", "Model Performance", "Prediction & Explainability", "Business Application"],
)
st.sidebar.caption("SVC + Genetic Optimization · Telco Customer Churn")
st.sidebar.caption("Demostración académica · Datos históricos")


if pagina == "Inicio":
    st.subheader("Panel ejecutivo")
    n = len(df)
    abandonos = int((df["Churn"] == "Yes").sum())
    a, b, c, d = st.columns(4)
    a.metric("Clientes históricos", f"{n:,}")
    b.metric("Abandonos registrados", f"{abandonos:,}")
    c.metric("Tasa de abandono", f"{100 * abandonos/n:.2f}%")
    d.metric("Cargo mensual medio", f"${df['MonthlyCharges'].mean():.2f}")
    a, b = st.columns(2)
    with a:
        datos = df.groupby(["Contract", "Churn"]).size().reset_index(name="Clientes")
        st.plotly_chart(px.bar(datos, x="Contract", y="Clientes", color="Churn", barmode="group",
                               title="Abandono por contrato"), use_container_width=True)
    with b:
        tasas = df.groupby("InternetService")["Churn"].apply(
            lambda s: 100 * s.eq("Yes").mean()).reset_index(name="Tasa (%)")
        st.plotly_chart(px.bar(tasas, x="InternetService", y="Tasa (%)", color="InternetService",
                               title="Abandono histórico por servicio"), use_container_width=True)
    st.subheader("De la predicción a la acción")
    a, b, c = st.columns(3)
    for box, title, descr in zip(
        [a, b, c], ["01 · Detecta", "02 · Prioriza", "03 · Actúa"],
        ["Calcula una puntuación de abandono con SVC.",
         "Ordena tu cartera según el score y la capacidad comercial.",
         "Exporta contactos y evalúa escenarios económicos."],
    ):
        box.markdown(f'<div class="brand-card"><h4>{title}</h4><p>{descr}</p></div>',
                     unsafe_allow_html=True)
    st.caption("Los KPI describen el dataset histórico, no una cartera real de una empresa conectada.")


elif pagina == "EDA":
    st.header("📊 Exploratory Data Analysis")
    st.write("Explora segmentos de clientes y observa su abandono histórico.")
    c1, c2 = st.columns(2)
    contrato = c1.selectbox("Tipo de contrato", ["Todos"] + sorted(df["Contract"].dropna().unique().tolist()))
    internet = c2.selectbox("Servicio de internet", ["Todos"] + sorted(df["InternetService"].dropna().unique().tolist()))

    filtrado = df.copy()
    if contrato != "Todos":
        filtrado = filtrado[filtrado["Contract"] == contrato]
    if internet != "Todos":
        filtrado = filtrado[filtrado["InternetService"] == internet]
    if filtrado.empty:
        st.warning("No hay clientes con esa combinación de filtros.")
        st.stop()

    total = len(filtrado)
    churn = int((filtrado["Churn"] == "Yes").sum())
    a, b, c, d = st.columns(4)
    a.metric("Clientes", f"{total:,}")
    b.metric("Abandonaron", f"{churn:,}")
    c.metric("Tasa de abandono", f"{100 * churn / total:.2f}%")
    d.metric("Cargo mensual medio", f"${filtrado['MonthlyCharges'].mean():.2f}")

    left, right = st.columns(2)
    with left:
        fig = px.histogram(filtrado, x="Churn", color="Churn", title="Distribución de abandono")
        st.plotly_chart(fig, use_container_width=True)
        fig = px.histogram(filtrado, x="Contract", color="Churn", barmode="group", title="Abandono por contrato")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.box(filtrado, x="Churn", y="MonthlyCharges", color="Churn", title="Cargos mensuales")
        st.plotly_chart(fig, use_container_width=True)
        fig = px.histogram(filtrado, x="tenure", color="Churn", nbins=30, barmode="overlay", title="Antigüedad (meses)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tabla de clientes")
    st.dataframe(filtrado, use_container_width=True, hide_index=True)


elif pagina == "Centro de retención":
    st.header("🎯 Centro de retención")
    st.write("Convierte una cartera CSV en una lista de contactos priorizados y un escenario económico editable.")
    modelo = obtener_modelo()
    archivo = st.file_uploader("Subir cartera de clientes (CSV)", type=["csv"])
    st.download_button("Descargar plantilla CSV de entrada", descargar_csv(
        df.head(1).drop(columns=["customerID", "Churn"])),
        "plantilla_clientes.csv", "text/csv")
    if archivo is None:
        st.info("Modo demostración: 200 clientes históricos. Para analizar una cartera nueva, sube un CSV con las 19 columnas de la plantilla.")
        cartera = df.head(200).drop(columns=["Churn"]).copy()
        id_cartera = ("demo_200", MODEL_PATH.stat().st_mtime_ns)
        modo_demo = True
    else:
        try:
            contenido = archivo.getvalue()
            cartera = pd.read_csv(archivo)
        except Exception as exc:
            st.error(f"No se pudo leer el CSV: {exc}")
            st.stop()
        id_cartera = (sha256(contenido).hexdigest(), MODEL_PATH.stat().st_mtime_ns)
        modo_demo = False

    cartera = normalizar_vacios(cartera)
    errores = validar_cartera(cartera, columnas_entrada)
    if errores:
        for error in errores:
            st.error(error)
        st.stop()
    if "customerID" in cartera and cartera["customerID"].duplicated().any():
        st.warning("Hay identificadores customerID duplicados. Comprueba si pertenecen al mismo cliente antes de contactar.")
    vacios = cartera[list(columnas_entrada)].isna().sum()
    if vacios.sum() > 0:
        detalle = ", ".join(f"{col}: {int(n)}" for col, n in vacios.items() if n)
        st.warning(f"Se encontraron {int(vacios.sum())} valores vacíos ({detalle}). "
                   "Se procesarán mediante la imputación del pipeline entrenado.")

    if st.session_state.get("id_cartera") != id_cartera:
        st.session_state.pop("cartera_puntuada", None)
        st.session_state["id_cartera"] = id_cartera

    st.caption(f"Cartera lista: {len(cartera):,} filas. No se envían datos a servicios externos desde este código.")
    if st.button("Analizar cartera", type="primary"):
        try:
            with st.spinner("Prediciendo riesgo de abandono..."):
                datos = cartera.drop(columns=["Churn"], errors="ignore")
                st.session_state["cartera_puntuada"] = puntuar_cartera(datos, modelo, columnas_entrada)
                st.session_state["modo_demo"] = modo_demo
        except Exception as exc:
            st.error(f"Error al analizar la cartera: {exc}")

    if "cartera_puntuada" in st.session_state:
        resultados = st.session_state["cartera_puntuada"]
        if st.session_state.get("modo_demo"):
            st.warning("DEMO: estos clientes históricos pueden haber participado en el entrenamiento. No utilices estas predicciones para reportar métricas de evaluación.")
        a, b, c = st.columns(3)
        a.metric("Clientes evaluados", f"{len(resultados):,}")
        b.metric("Prioridad alta (≥70 %)", int(resultados["P_Abandono"].ge(0.7).sum()))
        c.metric("Predicción Churn", int(resultados["Prediccion"].eq("Churn").sum()))
        a, b = st.columns(2)
        with a:
            st.plotly_chart(px.histogram(resultados, x="P_Abandono", nbins=20,
                                          title="Distribución de puntuaciones"), use_container_width=True)
        with b:
            conteo = resultados["Prioridad"].value_counts().rename_axis("Prioridad").reset_index(name="Clientes")
            st.plotly_chart(px.bar(conteo, x="Prioridad", y="Clientes", color="Prioridad",
                                    title="Cartera por prioridad"), use_container_width=True)

        st.divider()
        st.subheader("Configura tu campaña")
        capacidad = st.slider("Capacidad del equipo: contactos por campaña", 1, len(resultados), min(30, len(resultados)))
        a, b, c = st.columns(3)
        valor = a.number_input("Valor neto supuesto por cliente retenido ($)", min_value=0.0, value=120.0, step=10.0)
        costo = b.number_input("Costo supuesto por contacto ($)", min_value=0.0, value=8.0, step=1.0)
        eficacia = c.slider("Efectividad hipotética de la oferta (%)", 0, 100, 20) / 100
        criterio = st.radio("Ordenar contactos por", ["Mayor riesgo", "Mayor valor esperado"], horizontal=True,
                           help="El segundo criterio incorpora los supuestos económicos. No representa un valor comprobado.")
        ordenados, contactos, beneficio = evaluar_campana(resultados, capacidad, eficacia, valor, costo, criterio)
        monto_contactos = capacidad * costo
        potenciales = float((contactos["P_Abandono"] * eficacia).sum())
        a, b, c = st.columns(3)
        a.metric("Contactos seleccionados", f"{len(contactos):,}")
        b.metric("Costo de la campaña (supuesto)", f"${monto_contactos:,.2f}")
        c.metric("Balance esperado (simulación)", f"${beneficio:,.2f}")
        st.caption(f"Clientes potencialmente retenidos (modelo hipotético): {potenciales:.1f}. "
                   "Se calcula como suma de (score × efectividad) entre los contactados.")
        if beneficio < 0:
            st.warning("Bajo estos supuestos la campaña tendría balance negativo. Revisa costos, efectividad o capacidad.")
        else:
            st.success("El escenario elegido presenta balance positivo bajo los supuestos indicados; no es una garantía de retorno.")
        st.subheader("Lista operativa de contactos")
        columnas_tabla = [c for c in ("customerID", "P_Abandono", "Prioridad", "Accion sugerida", "Contract",
                                      "tenure", "MonthlyCharges", "Valor esperado ($)") if c in contactos.columns]
        st.dataframe(contactos[columnas_tabla], hide_index=True, use_container_width=True)
        a, b = st.columns(2)
        a.download_button("⬇️ Descargar contactos seleccionados", descargar_csv(contactos),
                          "contactos_nexoretain.csv", "text/csv", type="primary")
        b.download_button("Descargar cartera puntuada completa", descargar_csv(ordenados),
                          "cartera_puntuada.csv", "text/csv")
        st.caption("Fórmula ilustrativa por contacto: P(Churn) × efectividad supuesta × valor neto supuesto − costo de contacto. "
                   "Los scores pueden no estar calibrados; evitar abandonos requiere medir resultados con un grupo de control. "
                   "La app no registra campañas, no almacena una base de datos de clientes ni permite integración CRM todavía.")


elif pagina == "Model Performance":
    st.header("🔬 Laboratorio del modelo")
    st.write("Evidencia técnica para comparar calidad predictiva, costos de errores y esfuerzo de optimización.")
    st.info("Estos resultados corresponden a VALIDACIÓN y se muestran separados de cualquier evaluación de test final.")

    # Respaldo si aún no se ha exportado el CSV del notebook.
    metricas_respaldo = pd.DataFrame({
        "Modelo": ["SVC Base", "Genético V1", "Genético V2"],
        "Accuracy": [0.8091, 0.8084, 0.7700],
        "Precision": [0.6882, 0.7241, 0.5477],
        "Recall": [0.5134, 0.4492, 0.7674],
        "F1": [0.5881, 0.5545, 0.6392],
        "ROC-AUC": [0.8123, 0.8072, 0.8340],
    })
    metricas_archivo = cargar_csv_opcional("metricas_validacion.csv")
    columnas_metricas = ["Modelo", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    if metricas_archivo is not None and set(columnas_metricas).issubset(metricas_archivo.columns):
        metricas = metricas_archivo[columnas_metricas].copy()
        for columna in columnas_metricas[1:]:
            metricas[columna] = pd.to_numeric(metricas[columna], errors="coerce")
        metricas = metricas.dropna(subset=columnas_metricas).drop_duplicates("Modelo")
        if metricas.empty:
            metricas = metricas_respaldo
    else:
        metricas = metricas_respaldo
    matrices = {"SVC Base": [[948, 87], [182, 192]],
                "Genético V2": [[798, 237], [87, 287]]}
    base = resumen_matriz_confusion(matrices["SVC Base"])
    v2 = resumen_matriz_confusion(matrices["Genético V2"])

    tab1, tab2, tab3 = st.tabs(["Comparación de modelos", "Optimización", "Evaluación y límites"])
    with tab1:
        st.caption(f"{len(metricas)} modelos comparados en el mismo conjunto de validación. "
                   "Grid Search, Random Search, TPE y NSGA-II se leen de resultados/metricas_validacion.csv.")
        a, b, c, d = st.columns(4)
        a.metric("Recall V2", f"{v2['Recall']:.1%}", f"{(v2['Recall'] - base['Recall']) * 100:+.1f} puntos vs base")
        b.metric("Abandonos adicionales detectados", f"+{v2['VP'] - base['VP']}")
        c.metric("Falsas alarmas adicionales", f"+{v2['FP'] - base['FP']}", delta_color="inverse")
        d.metric("F1 de validación V2", "0.6392")

        st.dataframe(metricas.style.format({x: "{:.4f}" for x in metricas.columns if x != "Modelo"}),
                     hide_index=True, use_container_width=True)
        metrica = st.selectbox("Comparar métrica", ["F1", "Recall", "Precision", "Accuracy", "ROC-AUC"])
        fig = px.bar(metricas, x="Modelo", y=metrica, color="Modelo", text_auto=".4f",
                     title=f"Resultados de validación: {metrica}", range_y=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("¿Qué tipos de errores comete cada modelo?")
        c1, c2 = st.columns(2)
        with c1:
            mostrar_matriz(matrices["SVC Base"], "SVC Base (validación)")
        with c2:
            mostrar_matriz(matrices["Genético V2"], "SVC Genético V2 (validación)")
        st.caption("VP: abandono detectado; FN: abandono no detectado; FP: alerta para quien no abandonó; VN: cliente correctamente identificado como no abandono.")
        st.warning("V2 identifica 95 abandonos más, pero genera 150 falsas alarmas más en las mismas 1.409 observaciones de validación. "
                   "No es mejor en todas las métricas: baja Accuracy y Precision respecto al modelo base.")
        st.markdown("**Criterio de selección:** elegimos V2 para un escenario en el que detectar abandonos es prioritario. "
                    "La decisión empresarial debe contrastar el valor de recuperar clientes con el costo de intervenir sobre falsas alarmas.")

    with tab2:
        st.subheader("Evolución de la optimización genética")
        genetica = pd.DataFrame({
            "Experimento": ["V1", "V2"],
            "Generaciones": [5, 10], "Población": [6, 10],
            "F1 medio CV": [0.5895, 0.6225], "Tiempo (s)": [46.55, 166.32],
        })
        st.dataframe(genetica, hide_index=True, use_container_width=True)
        st.caption("F1 medio de validación cruzada (fitness), no F1 del conjunto de validación de la primera pestaña. "
                   "Los tiempos se midieron en ejecuciones diferentes y dependen del equipo.")
        st.write("**Mejor configuración V2:** `C = 0.3162`, `gamma = 0.3162`, `class_weight = balanced`.")
        historial = cargar_csv_opcional("fitness_ga_v2.csv")
        if historial is not None and {"Generacion", "F1"}.issubset(historial.columns):
            st.plotly_chart(px.line(historial, x="Generacion", y="F1", markers=True,
                                   title="Evolución registrada del fitness (F1)"), use_container_width=True)
        else:
            st.info("Para mostrar la evolución genética, exporta `resultados/fitness_ga_v2.csv` desde el notebook con columnas `Generacion` y `F1`.")

        st.subheader("Otras estrategias solicitadas en la tarea")
        comparacion = cargar_csv_opcional("comparacion_optimizacion.csv")
        if comparacion is not None:
            st.dataframe(comparacion, hide_index=True, use_container_width=True)
            st.caption("Resultados importados del notebook. Comprueba que todas las estrategias usaron particiones, métricas y presupuestos comparables.")
        else:
            st.warning("Pendiente: Grid Search, Random Search, TPE/Hyperband y optimización multiobjetivo (NSGA-II u otra). "
                       "Cuando estén ejecutados, exporta `resultados/comparacion_optimizacion.csv` para mostrarlos aquí.")
        pareto = cargar_csv_opcional("pareto_front.csv")
        if pareto is not None:
            numericas = pareto.select_dtypes(include="number").columns.tolist()
            if len(numericas) >= 2:
                st.plotly_chart(px.scatter(pareto, x=numericas[0], y=numericas[1],
                                           title="Resultados de optimización multiobjetivo"), use_container_width=True)
            else:
                st.info("El CSV de Pareto debe incluir al menos dos columnas numéricas de objetivos.")

    with tab3:
        st.subheader("Evaluación final independiente")
        test = cargar_csv_opcional("metricas_test.csv")
        if test is not None:
            st.dataframe(test, hide_index=True, use_container_width=True)
            st.caption("Métricas de test importadas desde el notebook; no deben utilizarse para elegir hiperparámetros.")
        else:
            st.warning("Falta `resultados/metricas_test.csv`. No mostramos métricas finales de prueba inventadas.")
        with st.expander("Transparencia y límites del modelo", expanded=True):
            st.markdown("""
            - Las cifras mostradas en las pestañas anteriores proceden de validación; el test final es independiente.
            - El SVC se optimizó con PyGAD para F1 de la clase Churn.
            - El modelo genera falsos positivos y falsos negativos; ambos tienen consecuencias comerciales.
            - Las probabilidades podrían no estar calibradas en una cartera de otra empresa.
            - Una evaluación histórica no demuestra ganancias ni retención incremental reales.
            - SHAP/LIME explican predicciones, pero no establecen causalidad.
            """)


elif pagina == "Prediction & Explainability":
    st.header("🧠 Predicción individual & explicabilidad")
    st.write("Evalúa un perfil individual y comprueba qué variables influyen en su puntuación de abandono.")
    modelo = obtener_modelo()

    # Usamos un registro real como plantilla de columnas y categorías.
    plantilla = df.iloc[0].drop(labels=["customerID", "Churn"]).to_dict()
    perfil = st.radio("Perfil de ejemplo", ["Personalizado", "Nuevo con contrato mensual", "Cliente de larga permanencia"],
                      horizontal=True, help="Son escenarios ilustrativos; la predicción exacta depende del modelo.")
    sugerencias = {
        "Personalizado": {},
        "Nuevo con contrato mensual": {"Contract": "Month-to-month", "tenure": 2, "MonthlyCharges": 85.0,
                                       "TotalCharges": 170.0, "InternetService": "Fiber optic"},
        "Cliente de larga permanencia": {"Contract": "Two year", "tenure": 60, "MonthlyCharges": 55.0,
                                          "TotalCharges": 3300.0, "InternetService": "DSL"},
    }
    inicial = {**plantilla, **sugerencias[perfil]}
    entrada = {}
    with st.form("form_cliente"):
        c1, c2 = st.columns(2)
        for i, columna in enumerate(plantilla):
            contenedor = c1 if i % 2 == 0 else c2
            with contenedor:
                if columna == "SeniorCitizen":
                    entrada[columna] = st.selectbox(columna, ["0", "1"],
                                                   index=int(str(inicial[columna]) in ("1", "1.0")))
                elif columna == "tenure":
                    entrada[columna] = st.number_input(columna + " (meses)", min_value=0, max_value=100,
                                                       value=int(inicial[columna]), step=1)
                elif columna in ["MonthlyCharges", "TotalCharges"]:
                    v = pd.to_numeric(inicial[columna], errors="coerce")
                    entrada[columna] = st.number_input(columna + " ($)", min_value=0.0,
                                                       value=float(v) if pd.notna(v) else 0.0, step=1.0)
                else:
                    opciones = sorted(df[columna].dropna().astype(str).unique().tolist())
                    elegido = str(inicial[columna])
                    entrada[columna] = st.selectbox(columna, opciones,
                                                   index=opciones.index(elegido) if elegido in opciones else 0)
        enviar = st.form_submit_button("Predecir abandono", type="primary")

    if enviar:
        try:
            cliente = preparar_clientes(pd.DataFrame([entrada]), columnas_entrada)
            probabilidad = float(modelo.predict_proba(cliente)[0, 1])
            prediccion = int(modelo.predict(cliente)[0])
            st.session_state["cliente_predicho"] = cliente
            st.session_state["probabilidad_predicha"] = probabilidad
            st.session_state["clase_predicha"] = prediccion
            st.session_state.pop("explicacion_lista", None)
        except Exception as exc:
            st.error(f"No se pudo predecir: {exc}")

    if "cliente_predicho" in st.session_state:
        cliente = st.session_state["cliente_predicho"]
        probabilidad = st.session_state["probabilidad_predicha"]
        prediccion = st.session_state["clase_predicha"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Probabilidad estimada de abandono", f"{100 * probabilidad:.2f}%")
        c2.metric("Clasificación del SVC", "Churn" if prediccion == 1 else "No Churn")
        c3.metric("Prioridad de contacto", "Alta" if probabilidad >= 0.7 else "Media" if probabilidad >= 0.4 else "Baja")
        st.progress(min(1.0, max(0.0, probabilidad)))
        st.caption("Estimación del modelo, no garantía de abandono. La clase predict() y la probabilidad pueden diferir cerca del umbral por la calibración interna del SVC.")
        st.info("Acción sugerida: " + ("contactar y revisar oferta de retención" if probabilidad >= 0.7 else "enviar al equipo para revisión" if probabilidad >= 0.4 else "seguimiento rutinario"))
        st.caption("La prioridad alta/media/baja es una regla operativa ilustrativa (umbrales 70 % y 40 %), no umbrales optimizados en test.")

        st.subheader("Explicación individual")
        st.write("SHAP y LIME explican esta predicción para la clase Churn = 1. Pueden tardar un momento.")
        if st.button("Calcular SHAP y LIME para este cliente"):
            try:
                import shap
                from lime.lime_tabular import LimeTabularExplainer

                prep = modelo.named_steps["prep"]
                svc = modelo.named_steps["svc"]
                columnas = prep.get_feature_names_out().tolist()
                muestra = preparar_clientes(df.drop(columns=["customerID", "Churn"]).sample(n=min(30, len(df)), random_state=42), columnas_entrada)
                fondo = pd.DataFrame(prep.transform(muestra), columns=columnas)
                fila = pd.DataFrame(prep.transform(cliente), columns=columnas)

                with st.spinner("Calculando SHAP..."):
                    explicador = shap.Explainer(lambda x: svc.predict_proba(x)[:, 1], fondo,
                                               algorithm="permutation", feature_names=columnas)
                    valores = explicador(fila, max_evals=2 * len(columnas) + 1)
                    contrib = agrupar_contribuciones_shap(columnas, valores.values[0], columnas_entrada)
                    contrib = contrib.loc[contrib["Contribución SHAP"].abs().nlargest(10).index]
                    graf = px.bar(contrib.sort_values("Contribución SHAP"), x="Contribución SHAP", y="Característica",
                                  orientation="h", title="SHAP: variables originales que influyen en el riesgo")
                    graf.update_traces(marker_color=["#F87171" if v > 0 else "#34D399" for v in
                                                     contrib.sort_values("Contribución SHAP")["Contribución SHAP"]])
                    st.plotly_chart(graf, use_container_width=True)
                    st.caption(f"Referencia SHAP: {float(np.ravel(valores.base_values)[0]):.3f}. "
                               "Rojo aumenta el riesgo; verde lo reduce respecto a la referencia.")

                with st.spinner("Calculando LIME..."):
                    numericas = {"tenure", "MonthlyCharges", "TotalCharges"}
                    cat_indices = [i for i, c in enumerate(columnas)
                                   if c.split("__", 1)[-1] not in numericas]
                    lime = LimeTabularExplainer(fondo.to_numpy(), feature_names=columnas,
                                                class_names=["No Churn", "Churn"],
                                                categorical_features=cat_indices,
                                                mode="classification", random_state=42)
                    exp = lime.explain_instance(fila.iloc[0].to_numpy(), svc.predict_proba,
                                                labels=(1,), num_features=10, num_samples=750)
                    reglas = pd.DataFrame(exp.as_list(label=1), columns=["Regla", "Contribución local"])
                    st.plotly_chart(px.bar(reglas.sort_values("Contribución local"), x="Contribución local", y="Regla",
                                           orientation="h", title="LIME local: clase Churn"), use_container_width=True)
                st.warning("SHAP/LIME muestran asociaciones en el modelo, no causas del abandono. "
                           "LIME perturba variables codificadas y podría formar perfiles artificiales. "
                           "Las explicaciones utilizan una muestra histórica de referencia y no justifican automáticamente una oferta.")
            except ImportError:
                st.error("Instala SHAP y LIME: python -m pip install shap lime")
            except Exception as exc:
                st.error(f"No se pudo generar la explicación: {exc}")


elif pagina == "Business Application":
    st.header("💼 NexoRetain | Caso de negocio")
    st.write("Una propuesta de inteligencia comercial para proveedores de telecomunicaciones: "
             "detectar clientes en riesgo, priorizar contactos y evaluar escenarios de retención.")
    st.info("Las métricas predictivas provienen del conjunto histórico de **validación**. "
            "Los precios, costos, retención lograda y beneficios que siguen son **hipótesis editables**, "
            "no resultados económicos observados.")

    simulador, producto, venta = st.tabs(["Simulador económico", "Producto y compradores", "Oferta y riesgos"])
    with simulador:
        st.subheader("¿En qué condiciones conviene utilizar el modelo V2?")
        st.caption("Comparación sobre 1.409 registros de validación. El escenario supone contactar "
                   "a todos los clientes clasificados como Churn por cada modelo.")
        c1, c2, c3, c4 = st.columns(4)
        valor = c1.number_input("Valor neto supuesto por abandono evitado ($)", min_value=0.0,
                                max_value=10000.0, value=200.0, step=25.0)
        costo = c2.number_input("Costo de cada contacto ($)", min_value=0.0,
                                max_value=500.0, value=8.0, step=1.0)
        eficacia = c3.slider("Efectividad hipotética de retención (%)", 0, 100, 25) / 100
        tarifa = c4.number_input("Costo adicional del servicio en este ciclo ($)",
                                 min_value=0.0, max_value=100000.0, value=300.0, step=50.0)

        base_cm = [[948, 87], [182, 192]]
        v2_cm = [[798, 237], [87, 287]]
        r = simular_diferencia_comercial(base_cm, v2_cm, eficacia, valor, costo, tarifa)
        a, b, c, d = st.columns(4)
        a.metric("Abandonos adicionales detectados", f"+{r['tp_extra']}")
        b.metric("Falsas alarmas adicionales", f"+{r['fp_extra']}", delta_color="inverse")
        c.metric("Contactos adicionales netos", f"+{r['contactos_extra']}")
        d.metric("Balance incremental hipotético", f"${r['balance']:,.2f}")

        st.markdown("**Cálculo del escenario:**")
        st.latex(r"\Delta B = 95\times e\times V - 245\times C - T")
        st.caption("e = efectividad hipotética de la campaña; V = valor neto supuesto de retener "
                   "un cliente; C = costo por contacto; T = costo adicional del servicio. "
                   "Incluimos los 245 contactos netos adicionales (95 VP + 150 FP), "
                   "no solamente las falsas alarmas.")

        if r["balance"] >= 0:
            st.success("Este escenario sería favorable con los supuestos seleccionados. "
                       "Debe comprobarse en un piloto real antes de justificar la compra.")
        else:
            st.warning("Con los supuestos seleccionados, el escenario no compensa los costos adicionales. "
                       "Un comprador debería exigir menor costo, mayor efectividad o un piloto controlado.")
        if r["tp_extra"] > 0 and valor > 0:
            punto = 100 * (r["contactos_extra"] * costo + tarifa) / (r["tp_extra"] * valor)
            st.metric("Efectividad mínima para alcanzar equilibrio", f"{punto:.1f}%" if punto <= 100 else ">100 %")
        else:
            st.caption("No se puede calcular un punto de equilibrio con valor neto supuesto igual a cero.")

        escenarios = pd.DataFrame({"Efectividad (%)": np.arange(0, 101, 5)})
        escenarios["Balance incremental ($)"] = [
            simular_diferencia_comercial(base_cm, v2_cm, e / 100, valor, costo, tarifa)["balance"]
            for e in escenarios["Efectividad (%)"]]
        fig = px.line(escenarios, x="Efectividad (%)", y="Balance incremental ($)", markers=True,
                      title="Sensibilidad: cuándo se recuperaría la inversión")
        fig.add_hline(y=0, line_dash="dash", line_color="#F0B56A")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("No extrapolamos automáticamente los resultados de 1.409 clientes a toda una operadora. "
                   "Las puntuaciones pueden no estar calibradas y detectar abandono no garantiza evitarlo.")

    with producto:
        st.subheader("Problema, compradores y solución")
        st.markdown("""
        **1. Problema de negocio.** Las operadoras pueden gastar recursos ofreciendo retención indiscriminada
        o reaccionar tarde ante clientes que abandonarán el servicio.

        **2. Producto.** NexoRetain es un prototipo de analítica predictiva y priorización comercial:
        permite cargar clientes, estimar riesgo de churn, revisar explicaciones SHAP/LIME y exportar listas.

        **3. Compradores potenciales.** Operadoras móviles, proveedores de internet y áreas de experiencia
        del cliente, especialmente equipos de retención con capacidad de contacto limitada.

        **4. Propuesta de valor.** Convertir datos de clientes en una lista de trabajo priorizada,
        con transparencia sobre errores y costos.

        **5. Valor del Machine Learning.** SVC Genético V2 alcanzó un Recall de **76.74 % en validación**
        frente a **51.34 % del SVC base**, aunque su Precision disminuyó de 68.82 % a 54.77 %.

        **6. Diferenciación propuesta.** Una experiencia que combina predicción por lotes,
        explicabilidad individual y evaluación económica del costo de los errores.
        No se afirma superioridad frente a soluciones comerciales externas sin una comparación directa.
        """)
        st.subheader("Cómo funcionaría un piloto")
        st.markdown("""
        1. **Preparar:** definir datos, consentimiento/base legal y variables disponibles.
        2. **Analizar:** puntuar una muestra de cartera y proponer contactos prioritarios.
        3. **Experimentar:** asignar aleatoriamente clientes elegibles a intervención y grupo de control.
        4. **Medir:** comparar retención real, costos e impacto incremental de la campaña.
        5. **Decidir:** contratar, ajustar o rechazar la solución según resultados documentados.
        """)

    with venta:
        st.subheader("Modelo comercial propuesto")
        planes = pd.DataFrame([
            {"Oferta": "Piloto de evaluación", "Precio hipotético": "$300 por ciclo", "Alcance": "Carga CSV, predicción y lista priorizada"},
            {"Oferta": "Servicio recurrente futuro", "Precio hipotético": "$800 por mes", "Alcance": "Soporte y análisis periódicos; requiere implementación"},
            {"Oferta": "Integración empresarial futura", "Precio hipotético": "A cotizar", "Alcance": "CRM, permisos, auditoría y monitoreo: no disponibles aún"},
        ])
        st.dataframe(planes, hide_index=True, use_container_width=True)
        st.caption("**7. Ingresos:** precios exclusivamente ilustrativos para la propuesta académica. "
                   "No se ha verificado demanda ni disposición real de pago.")
        st.subheader("Riesgos que debemos comunicar")
        st.markdown("""
        **8. Limitaciones y mitigaciones:**

        - **Falsas alarmas:** pueden generar contactos o descuentos innecesarios; controlar el costo por intervención.
        - **Abandonos no detectados:** algunos clientes no serán priorizados; mantener canales de atención generales.
        - **Cambio de población:** probar, recalibrar y monitorear antes de operar con una cartera nueva.
        - **Privacidad:** los datos reales requerirían controles de acceso, políticas de conservación y evaluación legal.
        - **Efectividad comercial:** validar las ofertas con un grupo de control; el modelo predice riesgo, no causalidad ni respuesta a incentivos.
        - **Madurez técnica:** esta versión es un MVP académico sin usuarios, CRM, auditoría ni despliegue empresarial seguro.
        """)
        st.subheader("Preguntas difíciles del comprador")
        with st.expander("¿Por qué seleccionó el SVC genético V2?"):
            st.write("Porque en validación detectó más abandonos y alcanzó mayor F1 que el SVC base. "
                     "El costo son 150 falsas alarmas adicionales; no todos los negocios preferirán ese intercambio.")
        with st.expander("¿Cómo puedo confiar en las predicciones?"):
            st.write("Se muestran matrices y métricas de validación, resultados separados de test, "
                     "y explicaciones SHAP/LIME. Antes de producción se requiere evaluar el modelo en la empresa compradora.")
        with st.expander("¿Qué pasa si no hay retorno económico?"):
            st.write("El piloto debe tener indicadores y un criterio de decisión acordados. "
                     "Si no hay mejora incremental suficiente, proponemos ajustar o no contratar.")
