# Detección de anomalías en tráfico de servicios backend

Trabajo práctico integrador de **Inteligencia Artificial** — Universidad de Palermo.

Este repositorio contiene una validación experimental de detección de anomalías en una serie temporal de tasa de solicitudes entrantes a una API. Se estudia su posible **aplicación** a plataformas de streaming: **los datos utilizados no proceden de una plataforma de streaming**.

## Dataset

- **Origen:** Microsoft, [Cloud Monitoring Dataset](https://github.com/microsoft/cloud-monitoring-dataset).
- **Archivo original:** [ecommerce-api-incoming-rps/api-01.csv](https://github.com/microsoft/cloud-monitoring-dataset/blob/master/data/ecommerce-api-incoming-rps/api-01.csv).
- **Copia incluida:** `data/api-01.csv`.
- **Campos:** `TimeStamp` (fecha UTC), `Value` (tasa de solicitudes entrantes), `Label` (anomalía de referencia: 0/1).
- Las etiquetas se utilizan **solo para evaluar**, no para entrenar el modelo no supervisado.
- **Licencia del dataset:** MIT. Aviso original en `data/LICENSE_Microsoft.txt`. Se conserva la atribución a Microsoft.

## Cómo ejecutarlo

Requiere Python 3.10 o superior. Desde la carpeta raíz:

```bash
python -m pip install -r requirements.txt
python main.py
```

No es necesario descargar ningún archivo adicional, utilizar AWS ni configurar una API: el CSV está dentro del repositorio.

El programa procesa los datos, calcula las tres alternativas de detección y crea:

- `resultados/metricas.csv`: matriz de confusión y métricas de los métodos.
- `resultados/resumen.json`: cantidad de observaciones, limpieza y partición.
- `graficos/comparacion_modelos.png`: precisión, recall y F1-score.
- `graficos/trafico_y_anomalias.png`: fragmento de tráfico con etiquetas reales y anomalías detectadas.
- `graficos/anomalias_por_mes.png`: distribución mensual de registros anómalos.

Estos archivos se vuelven a generar en cada ejecución. El código también imprime los resultados por consola.

## Método: qué hace el código

1. Lee `data/api-01.csv` y comprueba los campos, valores y etiquetas.
2. Convierte las fechas a UTC, ordena las observaciones y consolida registros con una misma fecha: promedio de `Value` y máximo de `Label`.
3. Reserva los **primeros 70 %** de los registros para construir los criterios de detección y los **últimos 30 %** para evaluar (partición cronológica).
4. Calcula la mediana histórica de `Value` para cada hora UTC **solo con el tramo inicial**.
5. Compara tres métodos: **umbral fijo de solicitudes**, **desviación absoluta frente a la mediana horaria** y **Isolation Forest** (scikit-learn).
6. El modelo usa seis características: valor, desviación horaria, diferencia con la medición anterior, diferencia con 24 registros anteriores y hora codificada con seno/coseno.
7. A cada método se le asigna un umbral definido por el **percentil 98 de su puntuación en la partición inicial**, sin mirar las etiquetas de prueba.
8. Compara las predicciones con `Label` mediante precisión, recall, F1, matriz de confusión y episodios detectados.

La configuración del modelo incluye `n_estimators=100`, `max_samples=256` y `random_state=42`. Las métricas pueden diferir de experimentos previos realizados con otras implementaciones o parámetros; se deben citar los resultados emitidos por **este mismo script**.

## Qué significan los resultados

- **Verdadero positivo:** el modelo marcó anomalía y la etiqueta dice 1.
- **Falso positivo:** emitió una alerta sobre una lectura etiquetada como normal.
- **Falso negativo:** una lectura anómala no fue detectada.
- **Precisión:** TP / (TP + FP).
- **Recall:** TP / (TP + FN).
- **F1:** media armónica de precisión y recall.
- **Episodio detectado:** al menos una lectura del bloque continuo etiquetado como anomalía fue identificada.

La comparación es **exploratoria**, utiliza una sola serie histórica y no prueba que el sistema prediga incidentes ni que sirva sin ajustes en producción. La tasa de solicitudes, por sí sola, no permite inferir el estado de reproducción de video.

## Referencias

- [Microsoft, Cloud Monitoring Dataset](https://github.com/microsoft/cloud-monitoring-dataset).
- Liu, Ting y Zhou (2008), *Isolation Forest*. [DOI: 10.1109/ICDM.2008.17](https://doi.org/10.1109/ICDM.2008.17).
- [scikit-learn, IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).
- [Datadog, Anomaly Monitor](https://docs.datadoghq.com/monitors/types/anomaly/).
- [Amazon CloudWatch, Anomaly Detection](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/CloudWatch_Anomaly_Detection.html).
