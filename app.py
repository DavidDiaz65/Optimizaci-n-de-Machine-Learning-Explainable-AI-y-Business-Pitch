from pathlib import Path
from hashlib import sha256

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "svc_genetic_telco.pkl"
RESULTS_DIR = BASE_DIR / "resultados"

st.set_page_config(page_title="NexoRetain | Retención inteligente", page_icon="✦", layout="wide")
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = ["#2DD4BF", "#60A5FA", "#F59E0B", "#FB7185", "#A78BFA"]
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
 [data-testid="stTabs"] [role="tab"] {font-weight:650;}
 [data-testid="stDataFrame"] {border: 1px solid #2D4564; border-radius: 10px;}
 [data-testid="stMetricValue"] {color:#F6FBFF;}
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
    if criterio == "Mayor cargo mensual":
        if "MonthlyCharges" not in puntuados.columns:
            raise ValueError("Falta MonthlyCharges para priorizar por facturación.")
        columna = "MonthlyCharges"
    elif criterio == "Mayor riesgo":
        columna = "P_Abandono"
    elif criterio == "Mayor valor esperado":
        columna = "Valor esperado ($)"  # Coincide con riesgo si se usa el mismo valor para todos.
    else:
        raise ValueError(f"Criterio no reconocido: {criterio}")
    puntuados = puntuados.sort_values([columna, "P_Abandono"], ascending=False, kind="stable")
    contactos = puntuados.head(capacidad).copy()
    return puntuados, contactos, float(contactos["Valor esperado ($)"].sum())


def calcular_economia_cartera(contactos, eficacia, valor, costo, costo_fijo):
    """Proyección ilustrativa sobre scores, sin atribuir retención real."""
    if not 0 <= eficacia <= 1 or min(valor, costo, costo_fijo) < 0:
        raise ValueError("Los supuestos económicos no son válidos.")
    probabilidades = pd.to_numeric(contactos["P_Abandono"], errors="raise").to_numpy()
    if not np.isfinite(probabilidades).all() or ((probabilidades < 0) | (probabilidades > 1)).any():
        raise ValueError("Las puntuaciones deben estar entre 0 y 1.")
    retenidos_estimados = float(probabilidades.sum() * eficacia)
    ingreso_estimado = retenidos_estimados * valor
    costo_variable = len(contactos) * costo
    total_costos = costo_variable + costo_fijo
    umbral = total_costos / (probabilidades.sum() * valor) if probabilidades.sum() * valor else np.inf
    return {"retenidos": retenidos_estimados, "valor": ingreso_estimado,
            "costos": total_costos, "balance": ingreso_estimado - total_costos,
            "eficacia_equilibrio": umbral}


def proyectar_capacidades(cartera_ordenada, eficacia, valor, costo, costo_fijo):
    """Curva calculada de forma vectorizada para todas las capacidades posibles."""
    scores = cartera_ordenada["P_Abandono"].to_numpy(dtype=float)
    n = np.arange(1, len(scores) + 1, dtype=int)
    ingresos = np.cumsum(scores) * eficacia * valor
    gastos = n * costo + costo_fijo
    return pd.DataFrame({"Contactos": n, "Valor estimado ($)": ingresos,
                         "Costo total ($)": gastos, "Balance proyectado ($)": ingresos - gastos})


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
<p>Conoce el riesgo de abandono, organiza tu cartera y planifica campañas de retención.</p></div>""",
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
st.sidebar.caption("Inteligencia de retención · NexoRetain")
st.sidebar.caption("Modo demostración con datos históricos")


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
    st.header("📊 Exploración de clientes")
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
        criterio = st.radio("Ordenar contactos por", ["Mayor riesgo", "Mayor cargo mensual"], horizontal=True,
                           help="El cargo mensual es un dato registrado, no equivale al margen real del cliente.")
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
    st.header("📈 Rendimiento de NexoRetain")
    st.write("Conoce qué tan bien identifica clientes en riesgo y cuántas alertas necesitaría revisar el equipo.")
    st.caption("Resultados históricos de validación. El desempeño con una cartera nueva debe comprobarse antes de tomar decisiones reales.")

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

    tab1, tab2, tab3 = st.tabs(["Resumen de rendimiento", "Comparación de modelos", "Ficha técnica"])
    with tab1:
        st.subheader("Detección de abandono")
        a, b, c, d = st.columns(4)
        a.metric("Abandonos identificados", f"{v2['Recall']:.1%}",
                 f"{(v2['Recall'] - base['Recall']) * 100:+.1f} puntos frente al inicial")
        b.metric("Casos adicionales detectados", f"+{v2['VP'] - base['VP']}")
        c.metric("Alertas adicionales que no eran abandono", f"+{v2['FP'] - base['FP']}", delta_color="inverse")
        d.metric("Equilibrio de detección (F1)", "0.6392")

        st.write("En una muestra histórica con **374 clientes que sí abandonaron**, "
                 "NexoRetain V2 identificó **287**. El modelo inicial identificó **192**.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**De quienes realmente abandonaron**")
            graf = go.Figure()
            nombres = ["Modelo inicial", "NexoRetain V2"]
            graf.add_bar(name="Abandonos identificados", y=nombres, x=[base["VP"], v2["VP"]],
                         orientation="h", marker_color="#2DD4BF", text=[base["VP"], v2["VP"]],
                         textposition="inside")
            graf.add_bar(name="No detectados", y=nombres, x=[base["FN"], v2["FN"]],
                         orientation="h", marker_color="#FB7185", text=[base["FN"], v2["FN"]],
                         textposition="inside")
            graf.update_layout(barmode="stack", height=310, template="plotly_dark",
                               xaxis_title="Clientes que sí abandonaron", yaxis_title="",
                               legend=dict(orientation="h", y=-0.3),
                               margin=dict(l=20, r=10, t=20, b=60),
                               paper_bgcolor="#0B1324", plot_bgcolor="#0F1D32")
            st.plotly_chart(graf, use_container_width=True)
        with c2:
            st.markdown("**Alertas que recibiría el equipo comercial**")
            alertas = go.Figure()
            alertas.add_bar(name="Abandono real", y=nombres, x=[base["VP"], v2["VP"]],
                            orientation="h", marker_color="#2DD4BF", text=[base["VP"], v2["VP"]],
                            textposition="inside")
            alertas.add_bar(name="Alerta incorrecta", y=nombres, x=[base["FP"], v2["FP"]],
                            orientation="h", marker_color="#FBBF24", text=[base["FP"], v2["FP"]],
                            textposition="inside")
            alertas.update_layout(barmode="stack", height=310, template="plotly_dark",
                                  xaxis_title="Clientes señalados para contactar", yaxis_title="",
                                  legend=dict(orientation="h", y=-0.3),
                                  margin=dict(l=20, r=10, t=20, b=60),
                                  paper_bgcolor="#0B1324", plot_bgcolor="#0F1D32")
            st.plotly_chart(alertas, use_container_width=True)
        st.info("**Lectura comercial:** mayor cobertura de abandonos implica más oportunidades de contacto, "
                "pero también más casos que el equipo revisaría sin que existiera abandono. "
                "El costo real depende de cada campaña.")
        st.caption("Ambas comparaciones utilizan los mismos 1.409 registros de validación; no son resultados de una operadora conectada.")

    with tab2:
        st.subheader("Comparador de rendimiento")
        st.write("Explora siete alternativas evaluadas con la misma partición de validación.")
        etiquetas = {"F1": "Equilibrio entre detección y precisión (F1)",
                     "Recall": "Cobertura de abandonos (Recall)",
                     "Precision": "Precisión de las alertas (Precision)",
                     "Accuracy": "Aciertos generales (Accuracy)",
                     "ROC-AUC": "Capacidad de discriminación (ROC-AUC)"}
        seleccion = st.selectbox("Indicador que deseas comparar", list(etiquetas),
                                format_func=lambda clave: etiquetas[clave])
        orden = metricas.sort_values(seleccion, ascending=True).copy()
        colores = ["#2DD4BF" if m == "Genético V2" else "#55759A" for m in orden["Modelo"]]
        fig = go.Figure(go.Bar(
            x=orden[seleccion], y=orden["Modelo"], orientation="h", marker_color=colores,
            text=orden[seleccion].map(lambda v: f"{v:.1%}"), textposition="outside",
            customdata=orden[["Precision", "Recall", "F1"]].to_numpy(),
            hovertemplate="%{y}<br>Indicador: %{x:.1%}<br>Precisión: %{customdata[0]:.1%}"
                          "<br>Detección: %{customdata[1]:.1%}<br>F1: %{customdata[2]:.3f}<extra></extra>"))
        fig.update_layout(template="plotly_dark", height=430, showlegend=False,
                          xaxis=dict(title="Resultado en validación", range=[0, 1.08], tickformat=".0%"),
                          yaxis_title="", margin=dict(l=20, r=45, t=20, b=50),
                          paper_bgcolor="#0B1324", plot_bgcolor="#0F1D32")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Verde: motor predictivo utilizado por NexoRetain. "
                   "El indicador superior varía según la métrica: no existe una única alternativa ganadora en todo.")
        with st.expander("Ver tabla completa de los siete modelos"):
            st.dataframe(metricas.style.format({x: "{:.3f}" for x in metricas.columns if x != "Modelo"}),
                         hide_index=True, use_container_width=True)
        st.markdown("**Cómo interpretar los resultados**")
        a, b, c = st.columns(3)
        a.write("**Cobertura (Recall):** de los clientes que abandonaron, cuántos identificamos.")
        b.write("**Precisión:** de las alertas emitidas, cuántas correspondían a abandono real.")
        c.write("**F1:** resume ambas métricas para ayudar a comparar alternativas.")

    with tab3:
        st.subheader("Evaluación independiente")
        test = cargar_csv_opcional("metricas_test.csv")
        if test is not None and {"Modelo", "Recall", "F1"}.issubset(test.columns):
            fila_test = test.iloc[0]
            a, b = st.columns(2)
            a.metric("Cobertura en prueba final", f"{float(fila_test['Recall']):.1%}")
            b.metric("F1 en prueba final", f"{float(fila_test['F1']):.3f}")
            with st.expander("Ver todas las métricas de prueba"):
                st.dataframe(test, hide_index=True, use_container_width=True)
        else:
            st.info("No está disponible el archivo de evaluación final.")
        st.caption("La prueba final se mantiene separada de los resultados usados para seleccionar el modelo.")

        with st.expander("Proceso de entrenamiento y optimización", expanded=False):
            st.markdown("**Optimización genética del modelo final**")
            genetica = pd.DataFrame({
                "Experimento": ["V1", "V2"],
                "Generaciones": [5, 10], "Población": [6, 10],
                "F1 medio CV": [0.5895, 0.6225], "Tiempo (s)": [46.55, 166.32],
            })
            st.dataframe(genetica, hide_index=True, use_container_width=True)
            st.caption("F1 medio de validación cruzada y tiempos medidos en ejecuciones distintas.")
            st.write("**Parámetros del motor final:** C = 0.3162, gamma = 0.3162, class_weight = balanced.")
            historial = cargar_csv_opcional("fitness_ga_v2.csv")
            if historial is not None and {"Generacion", "F1"}.issubset(historial.columns):
                valores = pd.to_numeric(historial["F1"], errors="coerce").dropna()
                if valores.nunique() <= 1 and not valores.empty:
                    st.info(f"El mejor F1 registrado se mantuvo en {valores.iloc[0]:.4f} en todas las generaciones. "
                            "El historial no demuestra una mejora progresiva, por lo que omitimos una línea plana.")
                elif not valores.empty:
                    fig_fit = px.line(historial, x="Generacion", y="F1", markers=True,
                                      title="Mejor resultado registrado por generación")
                    fig_fit.update_layout(yaxis_title="F1 medio (CV)")
                    st.plotly_chart(fig_fit, use_container_width=True)
                st.dataframe(historial, hide_index=True, use_container_width=True)
            comparacion = cargar_csv_opcional("comparacion_optimizacion.csv")
            if comparacion is not None:
                st.markdown("**Otras estrategias de búsqueda**")
                st.dataframe(comparacion, hide_index=True, use_container_width=True)

        with st.expander("Explorar soluciones de optimización multiobjetivo", expanded=False):
            pareto = cargar_csv_opcional("pareto_front.csv")
            if pareto is not None and {"F1", "Recall"}.issubset(pareto.columns):
                frente = px.scatter(pareto, x="Recall", y="F1", hover_data=[c for c in ["C", "gamma"] if c in pareto],
                                    title="Alternativas evaluadas: cobertura frente a F1")
                frente.update_traces(marker=dict(size=13, color="#2DD4BF"))
                frente.update_layout(xaxis_title="Cobertura (Recall)", yaxis_title="Equilibrio (F1)",
                                     paper_bgcolor="#0B1324", plot_bgcolor="#0F1D32")
                st.plotly_chart(frente, use_container_width=True)
                st.caption("Cada punto es una configuración registrada durante la búsqueda multiobjetivo.")
            else:
                st.info("No hay datos suficientes para explorar la optimización multiobjetivo.")

        with st.expander("Condiciones y límites de interpretación", expanded=False):
            st.markdown("""
            - Los indicadores provienen de datos históricos y pueden cambiar en otra cartera.
            - Existen alertas incorrectas y abandonos que el modelo no detectará.
            - La probabilidad estimada puede requerir calibración antes de su aplicación comercial.
            - Identificar clientes en riesgo no demuestra que una oferta consiga retenerlos.
            - Las explicaciones locales del modelo no prueban relaciones causales.
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
    st.header("💼 Planificador comercial")
    st.write("Calcula el presupuesto de una campaña y descubre cuántos clientes puede atender tu equipo.")

    cartera_activa = st.session_state.get("cartera_puntuada")
    if isinstance(cartera_activa, pd.DataFrame) and not cartera_activa.empty:
        datos_plan = cartera_activa.copy()
        st.success(f"Cartera activa: {len(datos_plan):,} clientes analizados en el Centro de retención.")
    else:
        st.info("Vista de demostración: analizaremos 200 registros históricos. "
                "Para planificar con tu propio CSV, impórtalo antes en el Centro de retención.")
        try:
            modelo = obtener_modelo()
            datos_plan = puntuar_cartera(df.head(200).drop(columns=["Churn"]), modelo, columnas_entrada)
        except Exception as exc:
            st.error(f"No se pudo preparar la cartera de demostración: {exc}")
            st.stop()

    st.caption("La cartera y sus estimaciones se procesan en esta sesión. "
               "Las cantidades económicas son escenarios configurables, no ingresos comprobados.")

    st.subheader("Configuración de campaña")
    c1, c2, c3 = st.columns(3)
    capacidad = c1.slider("Contactos que puede realizar el equipo", min_value=1,
                          max_value=len(datos_plan), value=min(50, len(datos_plan)),
                          help="Número máximo de clientes a contactar en este ciclo.", key="com_capacidad")
    criterio = c2.selectbox("Criterio de priorización", ["Mayor riesgo", "Mayor cargo mensual"],
                            key="com_criterio")
    eficacia_pct = c3.slider("Efectividad estimada de la campaña (%)", 0, 100, 25,
                              help="Supuesto editable: proporción de posibles abandonos que se evitarían con la intervención.",
                              key="com_eficacia")

    c1, c2, c3 = st.columns(3)
    valor = c1.number_input("Valor neto por cliente retenido ($)", min_value=0.0,
                            max_value=100000.0, value=200.0, step=25.0, key="com_valor")
    costo = c2.number_input("Costo por contacto ($)", min_value=0.0,
                            max_value=10000.0, value=8.0, step=1.0, key="com_costo")
    costo_fijo = c3.number_input("Costo fijo de la campaña ($)", min_value=0.0,
                                 max_value=1000000.0, value=300.0, step=50.0, key="com_fijo")
    eficacia = eficacia_pct / 100

    ordenados, seleccionados, _ = evaluar_campana(datos_plan, capacidad, eficacia, valor, costo, criterio)
    economia = calcular_economia_cartera(seleccionados, eficacia, valor, costo, costo_fijo)
    a, b, c, d = st.columns(4)
    a.metric("Contactos planificados", f"{capacidad:,}")
    b.metric("Costo total presupuestado", f"${economia['costos']:,.0f}")
    c.metric("Retenciones estimadas", f"{economia['retenidos']:.1f}")
    d.metric("Balance proyectado", f"${economia['balance']:,.0f}")

    tab_evolucion, tab_sensibilidad, tab_clientes = st.tabs(
        ["Presupuesto por capacidad", "Escenarios de campaña", "Contactos seleccionados"])

    with tab_evolucion:
        st.subheader("¿Cómo cambia el presupuesto si contactamos a más clientes?")
        st.write("Explora los valores con el cursor. La línea vertical indica la capacidad seleccionada.")
        curva = proyectar_capacidades(ordenados, eficacia, valor, costo, costo_fijo)
        grafico = go.Figure()
        for columna, color in [
            ("Valor estimado ($)", "#2DD4BF"),
            ("Costo total ($)", "#FB7185"),
            ("Balance proyectado ($)", "#60A5FA"),
        ]:
            grafico.add_trace(go.Scatter(x=curva["Contactos"], y=curva[columna],
                                         name=columna, mode="lines", line=dict(color=color, width=3),
                                         hovertemplate="%{x} contactos<br>$%{y:,.0f}<extra>" + columna + "</extra>"))
        grafico.add_vline(x=capacidad, line_dash="dot", line_color="#FBBF24",
                          annotation_text="Capacidad actual", annotation_position="top")
        grafico.add_hline(y=0, line_dash="dash", line_color="#94A3B8")
        grafico.update_layout(template="plotly_dark", height=450, hovermode="x unified",
                              xaxis_title="Clientes contactados", yaxis_title="Importe estimado ($)",
                              legend_title="Concepto", margin=dict(t=35, b=40),
                              paper_bgcolor="#0B1324", plot_bgcolor="#0F1D32")
        st.plotly_chart(grafico, use_container_width=True)
        if np.isfinite(economia["eficacia_equilibrio"]):
            umbral = 100 * economia["eficacia_equilibrio"]
            st.metric("Efectividad mínima estimada para cubrir costos", f"{umbral:.1f}%" if umbral <= 100 else ">100 %")
        else:
            st.caption("No se puede calcular el punto de equilibrio cuando el valor esperado de retención es cero.")

    with tab_sensibilidad:
        st.subheader("Mapa de escenarios")
        st.write("Cada celda muestra el balance estimado al cambiar simultáneamente los contactos y la efectividad.")
        tamaños = np.unique(np.linspace(1, len(ordenados), min(16, len(ordenados))).astype(int))
        eficacias = np.arange(0, 101, 5)
        acumuladas = np.cumsum(ordenados["P_Abandono"].to_numpy(dtype=float))
        mapa = np.array([
            [acumuladas[n - 1] * (e / 100) * valor - (n * costo + costo_fijo)
             for n in tamaños] for e in eficacias
        ])
        figura = go.Figure(go.Heatmap(
            z=mapa, x=tamaños, y=eficacias, colorscale="RdYlGn", zmid=0,
            colorbar=dict(title="Balance ($)"),
            hovertemplate="Contactos: %{x}<br>Efectividad: %{y}%<br>Balance: $%{z:,.0f}<extra></extra>"))
        figura.add_trace(go.Scatter(x=[capacidad], y=[eficacia_pct], mode="markers",
                                    name="Escenario elegido", marker=dict(size=16, color="#FFFFFF",
                                                                          symbol="circle-open", line=dict(width=3))))
        figura.update_layout(template="plotly_dark", height=480,
                             xaxis_title="Contactos por campaña", yaxis_title="Efectividad supuesta (%)",
                             paper_bgcolor="#0B1324", plot_bgcolor="#0F1D32",
                             legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(figura, use_container_width=True)
        st.caption("Colores cálidos indican balance estimado negativo; los verdes, positivo. "
                   "El anillo blanco representa la configuración actual.")

    with tab_clientes:
        st.subheader("Lista preparada para el equipo de retención")
        columnas_vista = [c for c in ["customerID", "P_Abandono", "Prioridad", "Accion sugerida",
                                        "Contract", "tenure", "MonthlyCharges", "Valor esperado ($)"]
                           if c in seleccionados.columns]
        st.dataframe(seleccionados[columnas_vista], hide_index=True, use_container_width=True)
        st.download_button("⬇️ Descargar contactos de esta campaña", descargar_csv(seleccionados),
                           "campana_nexoretain.csv", "text/csv", type="primary")
        resumen = pd.DataFrame([{
            "Contactos": capacidad, "Efectividad supuesta (%)": eficacia_pct,
            "Valor por retención ($)": valor, "Costo por contacto ($)": costo,
            "Costo fijo ($)": costo_fijo, "Retenciones estimadas": round(economia["retenidos"], 2),
            "Costo total ($)": round(economia["costos"], 2),
            "Balance proyectado ($)": round(economia["balance"], 2),
        }])
        st.download_button("Descargar resumen económico", descargar_csv(resumen),
                           "resumen_campana_nexoretain.csv", "text/csv")

    with st.expander("Interpretación y condiciones de uso"):
        st.write("Este cálculo combina las puntuaciones de abandono con una efectividad comercial introducida "
                 "manualmente. Los scores pueden no estar calibrados para una nueva empresa. "
                 "La efectividad real y el impacto incremental deben medirse con campañas y grupos de control. "
                 "La aplicación todavía no realiza envíos automáticos, integración CRM ni seguimiento de resultados.")
