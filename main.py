"""TP Integrador: detección de anomalías en tráfico de servicios backend.

Fuente del dataset: Microsoft Cloud Monitoring Dataset (licencia MIT).
Los datos corresponden a una API de comercio electrónico, NO a streaming.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "api-01.csv"
RESULTS = ROOT / "resultados"
FIGURES = ROOT / "graficos"
RANDOM_STATE = 42
TRAIN_FRACTION = 0.70
THRESHOLD_QUANTILE = 0.98


def read_data(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path)
    required = {"TimeStamp", "Value", "Label"}
    if set(raw.columns) != required:
        raise ValueError(f"Columnas inesperadas: {list(raw.columns)}")
    raw["TimeStamp"] = pd.to_datetime(raw["TimeStamp"], utc=True, errors="raise")
    raw["Value"] = pd.to_numeric(raw["Value"], errors="raise")
    raw["Label"] = pd.to_numeric(raw["Label"], errors="raise")
    if raw.isna().any().any() or not raw["Label"].isin([0, 1]).all():
        raise ValueError("Valores faltantes o etiquetas no binarias")
    if not np.isfinite(raw["Value"].to_numpy()).all():
        raise ValueError("Valores numéricos no finitos")

    duplicate_count = int(raw["TimeStamp"].duplicated().sum())
    # Agregación explícita de timestamps duplicados; nunca se usa Label para construir features.
    clean = (raw.groupby("TimeStamp", as_index=False)
             .agg(Value=("Value", "mean"), Label=("Label", "max"))
             .sort_values("TimeStamp").reset_index(drop=True))
    metadata = {"registros_originales": len(raw), "duplicados": duplicate_count,
                "registros_limpios": len(clean),
                "anomalias_etiquetadas": int(clean["Label"].sum())}
    return clean, metadata


def features(frame: pd.DataFrame, hourly_reference: np.ndarray) -> np.ndarray:
    values = frame["Value"].to_numpy(dtype=float)
    hour = frame["TimeStamp"].dt.hour.to_numpy(dtype=int)
    previous = np.r_[values[0], values[:-1]]
    # Las 24 posiciones previas representan 24 horas en la serie casi horaria,
    # no una garantía de diferencia cronológica exacta si hay intervalos irregulares.
    previous_24 = np.r_[np.repeat(values[0], 24), values[:-24]]
    return np.column_stack((
        values,
        np.abs(values - hourly_reference[hour]),
        np.abs(values - previous),
        np.abs(values - previous_24),
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
    ))


def event_detection(labels: np.ndarray, predictions: np.ndarray) -> tuple[int, int]:
    starts = np.flatnonzero((labels == 1) & (np.r_[0, labels[:-1]] == 0))
    ends = np.flatnonzero((labels == 1) & (np.r_[labels[1:], 0] == 0))
    return int(sum(predictions[a:b + 1].any() for a, b in zip(starts, ends))), len(starts)


def evaluate(name: str, train_scores: np.ndarray, test_scores: np.ndarray,
             test_labels: np.ndarray) -> tuple[dict, np.ndarray]:
    # El umbral se aprende solamente sobre la porción histórica, nunca sobre test.
    threshold = float(np.quantile(train_scores, THRESHOLD_QUANTILE))
    pred = (test_scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_labels, pred, labels=[0, 1]).ravel()
    episodes_detected, episodes_total = event_detection(test_labels, pred)
    metrics = {"metodo": name, "umbral": threshold, "TP": int(tp), "FP": int(fp),
               "FN": int(fn), "TN": int(tn),
               "precision": float(precision_score(test_labels, pred, zero_division=0)),
               "recall": float(recall_score(test_labels, pred, zero_division=0)),
               "f1": float(f1_score(test_labels, pred, zero_division=0)),
               "episodios_detectados": episodes_detected, "episodios_totales": episodes_total}
    return metrics, pred


def generate_graphs(frame: pd.DataFrame, test: pd.DataFrame,
                    test_predictions: dict[str, np.ndarray], metrics: pd.DataFrame) -> None:
    FIGURES.mkdir(exist_ok=True, parents=True)

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    x = np.arange(len(metrics))
    width = 0.24
    for j, (key, label) in enumerate([("precision", "Precisión"),
                                      ("recall", "Recall"), ("f1", "F1")]):
        bars = ax.bar(x + (j - 1) * width, metrics[key].to_numpy() * 100, width, label=label)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xticks(x, ["Umbral fijo", "Patrón horario", "Isolation Forest"])
    ax.set_ylabel("Porcentaje (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Comparación experimental: conjunto de prueba")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "comparacion_modelos.png", dpi=170)
    plt.close(fig)

    by_month = (frame.assign(mes=frame["TimeStamp"].dt.strftime("%Y-%m"))
                .groupby("mes")["Label"].sum())
    fig, ax = plt.subplots(figsize=(9.5, 3.9))
    ax.bar(by_month.index, by_month.to_numpy())
    ax.set_title("Registros etiquetados como anómalos por mes")
    ax.set_ylabel("Registros anómalos")
    ax.tick_params(axis="x", labelrotation=35)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "anomalias_por_mes.png", dpi=170)
    plt.close(fig)

    # Fragmento visual explicable: los primeros 14 días del conjunto de prueba.
    sample = test.iloc[:336].copy()
    pred = test_predictions["Isolation Forest"][:len(sample)].astype(bool)
    actual = sample["Label"].eq(1).to_numpy()
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(sample["TimeStamp"], sample["Value"], linewidth=1, label="Tasa de solicitudes")
    ax.scatter(sample.loc[actual, "TimeStamp"], sample.loc[actual, "Value"],
               s=37, marker="o", label="Anomalías etiquetadas")
    ax.scatter(sample.loc[pred, "TimeStamp"], sample.loc[pred, "Value"],
               s=58, marker="x", label="Alertas de Isolation Forest")
    ax.set_title("Tráfico y anomalías: primeros 14 días del conjunto de prueba")
    ax.set_ylabel("Solicitudes por segundo (según origen)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES / "trafico_y_anomalias.png", dpi=170)
    plt.close(fig)


def main() -> None:
    frame, details = read_data(DATA)
    cut = int(len(frame) * TRAIN_FRACTION)
    train, test = frame.iloc[:cut], frame.iloc[cut:]
    if not train["Label"].isin([0, 1]).all() or test["Label"].sum() == 0:
        raise ValueError("No hay anomalías etiquetadas en la partición de prueba")

    # Se calcula la mediana por hora sobre train, antes de transformar test.
    base_median = float(train["Value"].median())
    hourly_reference = (train.groupby(train["TimeStamp"].dt.hour)["Value"]
                        .median().reindex(range(24)).fillna(base_median).to_numpy())
    matrix = features(frame, hourly_reference)
    xtrain, xtest = matrix[:cut], matrix[cut:]
    labels = test["Label"].to_numpy(dtype=int)

    # sklearn devuelve puntuaciones MAYORES para observaciones más normales.
    # El signo negativo hace que una puntuación MAYOR indique más anomalía.
    model = IsolationForest(n_estimators=100, max_samples=256,
                            random_state=RANDOM_STATE, contamination="auto", n_jobs=1)
    model.fit(xtrain)
    scores = {
        "Umbral fijo": (xtrain[:, 0], xtest[:, 0]),
        "Patrón horario": (xtrain[:, 1], xtest[:, 1]),
        "Isolation Forest": (-model.score_samples(xtrain), -model.score_samples(xtest)),
    }
    rows, predictions = [], {}
    for name, (scores_train, scores_test) in scores.items():
        m, pred = evaluate(name, scores_train, scores_test, labels)
        rows.append(m)
        predictions[name] = pred

    metrics = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True, parents=True)
    metrics.to_csv(RESULTS / "metricas.csv", index=False, float_format="%.8f")
    details.update({
        "registros_train": len(train), "registros_test": len(test),
        "anomalias_train": int(train["Label"].sum()),
        "anomalias_test": int(test["Label"].sum()),
        "inicio_test": str(test["TimeStamp"].iloc[0]),
        "modelo": "sklearn.ensemble.IsolationForest",
        "n_estimators": 100, "max_samples": 256, "random_state": RANDOM_STATE,
        "percentil_umbral_train": 98,
    })
    (RESULTS / "resumen.json").write_text(
        json.dumps(details, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    generate_graphs(frame, test, predictions, metrics)

    print("\nDATOS")
    print(json.dumps(details, indent=2, ensure_ascii=False))
    print("\nEVALUACIÓN EN TEST")
    print(metrics[["metodo", "TP", "FP", "FN", "TN", "precision", "recall", "f1",
                   "episodios_detectados", "episodios_totales"]].to_string(index=False,
                     float_format=lambda x: f"{x:.4f}"))
    print("\nSe generaron archivos en resultados/ y graficos/.")


if __name__ == "__main__":
    main()
