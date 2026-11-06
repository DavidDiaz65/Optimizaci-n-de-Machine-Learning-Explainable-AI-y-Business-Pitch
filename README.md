# NexoRetain — proyecto final

Aplicación Streamlit de retención de clientes con SVC optimizado, explicabilidad SHAP/LIME y simulación comercial.

## Ejecutar en Windows (PowerShell)

Abre la terminal **en esta carpeta**, donde están `app.py`, `Tarea7.ipynb`, el CSV y el modelo:

```powershell
conda create -n nexoretain python=3.11 -y
conda activate nexoretain
python -m pip install -r requirements.txt
python -c "import sklearn; print(sklearn.__version__)"
python -m streamlit run app.py
```

La versión de `scikit-learn` **debe ser 1.7.2**, porque el archivo `svc_genetic_telco.pkl` se creó con esa versión. No utilices el entorno Conda `base` si tiene 1.8.0 o 1.9.0.

## Archivos

- `app.py`: aplicación web.
- `Tarea7.ipynb`: entrenamiento y evaluación de modelos, optimizadores y XAI.
- `svc_genetic_telco.pkl`: pipeline SVC final ya entrenado.
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: datos históricos de demostración.
- `resultados/`: resultados de optimización, validación y prueba.
- `.streamlit/config.toml`: colores coherentes con el diseño oscuro.

## Actualizar el modelo y los resultados

Abre `Tarea7.ipynb` con el **directorio de trabajo en esta misma carpeta** y ejecútalo de arriba abajo. El notebook guarda `svc_genetic_telco.pkl` y los CSV en `resultados/` relativos al directorio actual. En VS Code puedes seleccionar el entorno `nexoretain` como kernel. Comprueba `Path.cwd()` antes de ejecutarlo.

Las métricas de validación y test se mantienen separadas. El simulador comercial presenta supuestos, **no ganancias demostradas** ni retención incremental medida.
