from pathlib import Path

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
def cargar_modelo(path):
    return joblib.load(path)


def cargar_csv_opcional(nombre):
    path = RESULTS_DIR / nombre
    return pd.read_csv(path) if path.exists() else None


def obtener_modelo():
    if not MODEL_PATH.exists():
        st.error("Falta svc_genetic_telco.pkl. Expórtalo desde el notebook y cópialo aquí.")
        st.stop()
    try:
        modelo = cargar_modelo(MODEL_PATH)
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
    datos = datos.loc[:, columnas].copy()
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
    st.write("Carga una cartera, obtén predicciones por lotes y descarga una lista priorizada para tu equipo comercial.")
    modelo = obtener_modelo()
    archivo = st.file_uploader("Subir cartera de clientes (CSV)", type=["csv"])
    st.download_button("Descargar plantilla CSV de entrada", descargar_csv(
        df.head(1).drop(columns=["customerID", "Churn"])),
        "plantilla_clientes.csv", "text/csv")
    if archivo is None:
        st.info("Modo demostración: analizaremos 200 perfiles del dataset histórico. También puedes subir tu propia cartera con las 19 columnas predictoras.")
        cartera = df.head(200).drop(columns=["Churn"]).copy()
        id_cartera = "demo_200"
        modo_demo = True
    else:
        try:
            cartera = pd.read_csv(archivo)
        except Exception as exc:
            st.error(f"No se pudo leer el CSV: {exc}")
            st.stop()
        id_cartera = (archivo.name, archivo.size)
        modo_demo = False

    if cartera.empty:
        st.warning("El archivo no tiene filas.")
        st.stop()
    if len(cartera) > 10000:
        st.error("Por rendimiento, utiliza un máximo de 10.000 clientes por lote.")
        st.stop()
    faltantes = [c for c in columnas_entrada if c not in cartera.columns]
    if faltantes:
        st.error("Faltan columnas necesarias: " + ", ".join(faltantes))
        st.stop()

    if st.session_state.get("id_cartera") != id_cartera:
        st.session_state.pop("cartera_puntuada", None)
        st.session_state["id_cartera"] = id_cartera

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

        capacidad = st.slider("Capacidad del equipo: contactos a realizar", 1, len(resultados), min(30, len(resultados)))
        contactos = resultados.sort_values("P_Abandono", ascending=False).head(capacidad)
        st.dataframe(contactos, hide_index=True, use_container_width=True)
        st.download_button("⬇️ Descargar contactos priorizados", descargar_csv(contactos),
                           "contactos_nexoretain.csv", "text/csv", type="primary")
        with st.expander("Exportar toda la cartera puntuada"):
            st.download_button("Descargar resultados completos", descargar_csv(resultados),
                               "cartera_puntuada.csv", "text/csv")

        st.divider()
        st.subheader("Simulador económico de la cartera seleccionada")
        a, b, c = st.columns(3)
        valor = a.number_input("Valor neto de retener un cliente ($)", min_value=0.0, value=120.0, step=10.0)
        costo = b.number_input("Costo por contacto ($)", min_value=0.0, value=8.0, step=1.0)
        eficacia = c.slider("Efectividad hipotética de la campaña (%)", 0, 100, 20) / 100
        beneficio = float((contactos["P_Abandono"] * eficacia * valor - costo).sum())
        st.metric("Valor esperado hipotético", f"${beneficio:,.2f}")
        st.caption("Supuesto: score × eficacia × valor del cliente − costo por contacto. No es un ROI validado. Las probabilidades podrían no estar calibradas y la eficacia debe verificarse mediante campañas reales.")
        st.caption("Los CSV cargados se procesan en la sesión: esta app no tiene una base de datos ni guarda una lista persistente de clientes.")


elif pagina == "Model Performance":
    st.header("🔬 Laboratorio del modelo")
    st.info("Los resultados de comparación proceden de VALIDACIÓN; no son métricas del test final.")

    # Los números reflejan los resultados de validación aportados en el notebook.
    metricas = pd.DataFrame({
        "Modelo": ["SVC Base", "Genético V1", "Genético V2"],
        "Accuracy": [0.8091, 0.8084, 0.7700],
        "Precision": [0.6882, 0.7241, 0.5477],
        "Recall": [0.5134, 0.4492, 0.7674],
        "F1": [0.5881, 0.5545, 0.6392],
        "ROC-AUC": [0.8123, 0.8072, 0.8340],
    })
    st.dataframe(metricas.style.format({x: "{:.4f}" for x in metricas.columns if x != "Modelo"}),
                 hide_index=True, use_container_width=True)

    metrica = st.selectbox("Comparar métrica", ["F1", "Recall", "Precision", "Accuracy", "ROC-AUC"])
    fig = px.bar(metricas, x="Modelo", y=metrica, color="Modelo", text_auto=".4f",
                 title=f"Comparación en validación: {metrica}", range_y=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Matrices de confusión en validación")
    c1, c2 = st.columns(2)
    with c1:
        mostrar_matriz([[948, 87], [182, 192]], "SVC Base")
    with c2:
        mostrar_matriz([[798, 237], [87, 287]], "SVC Genético V2")

    st.subheader("Optimización genética")
    st.dataframe(pd.DataFrame({
        "Experimento": ["V1", "V2"],
        "Generaciones": [5, 10],
        "Población": [6, 10],
        "F1 medio CV": [0.5895, 0.6225],
        "Tiempo (s)": [46.55, 166.32],
    }), hide_index=True, use_container_width=True)
    st.caption("Los tiempos se midieron en ejecuciones diferentes y dependen del equipo.")
    st.write("**Mejor configuración V2:** C = 0.3162, gamma = 0.3162, class_weight = balanced.")
    st.success("V2 detectó 95 abandonos adicionales respecto al SVC base, pero también produjo 150 falsas alarmas adicionales en validación.")

    historial = cargar_csv_opcional("fitness_ga_v2.csv")
    if historial is not None and {"Generacion", "F1"}.issubset(historial.columns):
        st.plotly_chart(px.line(historial, x="Generacion", y="F1", markers=True,
                               title="Evolución de la mejor función fitness"), use_container_width=True)
    else:
        st.caption("Para visualizar la evolución genética, exporta resultados/fitness_ga_v2.csv desde el notebook.")

    test = cargar_csv_opcional("metricas_test.csv")
    if test is not None:
        st.subheader("Evaluación final en test (separada)")
        st.dataframe(test, hide_index=True, use_container_width=True)
    else:
        st.warning("Falta resultados/metricas_test.csv. No se muestran métricas finales inventadas.")

    with st.expander("Transparencia y límites del modelo"):
        st.markdown("""
        - La tabla compara modelos sobre validación; el test final debe reportarse por separado.
        - El SVC fue optimizado con PyGAD para F1 de la clase de abandono.
        - Las probabilidades no necesariamente están calibradas para una cartera nueva.
        - Las métricas de un dataset histórico no garantizan el mismo desempeño en otra empresa.
        - La importancia predictiva de una característica no demuestra causalidad.
        """)


elif pagina == "Prediction & Explainability":
    st.header("🧠 Predicción individual & explicabilidad")
    st.write("Introduce los datos de un cliente. El pipeline aplicará el preprocesamiento automáticamente.")
    modelo = obtener_modelo()

    # Usamos un registro real como plantilla de columnas y categorías.
    plantilla = df.iloc[0].drop(labels=["customerID", "Churn"]).to_dict()
    entrada = {}
    with st.form("form_cliente"):
        c1, c2 = st.columns(2)
        for i, columna in enumerate(plantilla):
            contenedor = c1 if i % 2 == 0 else c2
            with contenedor:
                if columna == "SeniorCitizen":
                    entrada[columna] = st.selectbox(columna, ["0", "1"])
                elif columna == "tenure":
                    entrada[columna] = st.number_input(columna + " (meses)", min_value=0, max_value=100, value=12, step=1)
                elif columna in ["MonthlyCharges", "TotalCharges"]:
                    entrada[columna] = st.number_input(columna + " ($)", min_value=0.0, value=70.0 if columna == "MonthlyCharges" else 840.0, step=1.0)
                else:
                    opciones = sorted(df[columna].dropna().astype(str).unique().tolist())
                    entrada[columna] = st.selectbox(columna, opciones)
        enviar = st.form_submit_button("Predecir abandono", type="primary")

    if enviar:
        try:
            cliente = preparar_clientes(pd.DataFrame([entrada]), columnas_entrada)
            probabilidad = float(modelo.predict_proba(cliente)[0, 1])
            prediccion = int(modelo.predict(cliente)[0])
            st.session_state["cliente_predicho"] = cliente
            st.session_state["probabilidad_predicha"] = probabilidad
            st.session_state["clase_predicha"] = prediccion
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

        st.subheader("Explicación individual")
        st.write("La generación de explicaciones puede tardar unos segundos. Se explicará la clase Churn = 1.")
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
                    contrib = pd.DataFrame({"Característica": columnas, "SHAP": valores.values[0]})
                    contrib = contrib.reindex(contrib["SHAP"].abs().sort_values(ascending=False).index).head(12)
                    st.plotly_chart(px.bar(contrib.sort_values("SHAP"), x="SHAP", y="Característica",
                                           orientation="h", title="SHAP local: contribución a P(Churn)"), use_container_width=True)
                    st.caption(f"Probabilidad de referencia SHAP: {float(valores.base_values[0]):.3f}")

                with st.spinner("Calculando LIME..."):
                    numericas = ["tenure", "MonthlyCharges", "TotalCharges"]
                    cat_indices = [i for i, c in enumerate(columnas) if c not in numericas]
                    lime = LimeTabularExplainer(fondo.to_numpy(), feature_names=columnas,
                                                class_names=["No Churn", "Churn"],
                                                categorical_features=cat_indices,
                                                mode="classification", random_state=42)
                    exp = lime.explain_instance(fila.iloc[0].to_numpy(), svc.predict_proba,
                                                labels=(1,), num_features=10, num_samples=1000)
                    reglas = pd.DataFrame(exp.as_list(label=1), columns=["Regla", "Contribución local"])
                    st.plotly_chart(px.bar(reglas.sort_values("Contribución local"), x="Contribución local", y="Regla",
                                           orientation="h", title="LIME local: clase Churn"), use_container_width=True)
                st.warning("SHAP/LIME explican el comportamiento del modelo, no causas. Las perturbaciones de columnas One-Hot pueden ser combinaciones artificiales. La muestra de referencia es demostrativa.")
            except ImportError:
                st.error("Instala SHAP y LIME: python -m pip install shap lime")
            except Exception as exc:
                st.error(f"No se pudo generar la explicación: {exc}")


elif pagina == "Business Application":
    st.header("💼 Impacto comercial & estrategia de retención")
    st.write("Simulación de la decisión de contactar clientes identificados con riesgo de abandono.")
    st.info("Los valores económicos son SUPUESTOS editables, no ingresos ni ahorros demostrados.")

    c1, c2, c3 = st.columns(3)
    beneficio = c1.number_input("Beneficio neto esperado por abandono evitado ($)", min_value=0.0, value=50.0, step=5.0)
    costo = c2.number_input("Costo por contactar una falsa alarma ($)", min_value=0.0, value=10.0, step=5.0)
    eficacia = c3.slider("Porcentaje de abandonos detectados que logra retener la campaña", 0, 100, 20, step=5) / 100

    # Diferencias observadas en validación: V2 detectó 95 verdaderos positivos más,
    # pero también marcó 150 falsos positivos adicionales respecto al SVC base.
    adicionales_tp = 95
    adicionales_fp = 150
    retenidos_esperados = adicionales_tp * eficacia
    valor = retenidos_esperados * beneficio - adicionales_fp * costo

    a, b, c = st.columns(3)
    a.metric("Abandonos adicionales detectados", adicionales_tp)
    b.metric("Falsas alarmas adicionales", adicionales_fp)
    c.metric("Valor incremental hipotético", f"${valor:,.2f}")

    st.write("**Modelo de simulación:** 95 × tasa de retención de la campaña × beneficio − 150 × costo de falsa alarma.")
    st.caption("Cálculo ilustrativo sobre 1.409 observaciones de validación. No representa una proyección validada para toda una empresa.")
    st.subheader("Sensibilidad económica")
    escenarios = pd.DataFrame({"Efectividad (%)": np.arange(0, 101, 5)})
    escenarios["Valor incremental ($)"] = (95 * escenarios["Efectividad (%)"] / 100 * beneficio - 150 * costo)
    fig = px.line(escenarios, x="Efectividad (%)", y="Valor incremental ($)", markers=True,
                  title="¿Cómo cambia el resultado con la efectividad de la campaña?")
    fig.add_hline(y=0, line_dash="dash", line_color="#F0B56A")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Plan comercial recomendado")
    st.markdown("""
    1. **Priorizar:** ordenar clientes por score e identificar segmentos que ameritan revisión.
    2. **Actuar:** aplicar una oferta a un grupo experimental y conservar otro como control.
    3. **Medir:** calcular retención incremental y costo real por cliente conservado.
    4. **Mejorar:** calibrar scores y actualizar el modelo cuando cambie el comportamiento de la cartera.
    """)
    st.subheader("Propuesta de valor")
    st.write("NexoRetain ayuda a priorizar acciones comerciales, comparar costos de errores de clasificación y comunicar predicciones mediante explicaciones de IA.")
    st.caption("MVP académico: faltan autenticación, almacenamiento seguro, integración CRM, auditoría, monitoreo y pruebas con clientes reales antes de su uso empresarial en producción.")
