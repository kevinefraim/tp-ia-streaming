"""Experimento reproducible: anomalías de audiencia en transmisiones en vivo.

Fuente real: YTLive (Mozhganfar et al., NOMS 2026).
Etiquetas de ataques: SOLO SINTÉTICAS. No se identifican bots reales.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "origen"
OUTPUT = ROOT / "resultados"
FIG = ROOT / "graficos"
SOURCE = "https://raw.githubusercontent.com/ghalandar/YTLive/a9ce70a470ddfdd52619a4f110bd33a8fcc1d5df/viewers.csv"
SEED = 42
WINDOW = 8                  # 8 mediciones a intervalos ~5 min
TARGET_STREAMS = 180
TRAIN_SHARE = .70
QUANTILE = .95


def download_source() -> Path:
    CACHE.mkdir(exist_ok=True, parents=True)
    out = CACHE / "viewers.csv"
    if out.is_file() and out.stat().st_size > 10_000_000:
        return out
    req = Request(SOURCE, headers={"User-Agent": "TP-IA-university-research/1.0"})
    with urlopen(req, timeout=120) as f, out.open("wb") as w:
        while block := f.read(1024 * 1024):
            w.write(block)
    if out.stat().st_size < 10_000_000:
        raise RuntimeError("Descarga incompleta de YTLive")
    return out


def read_observations(path: Path) -> pd.DataFrame:
    """Leer CSV ancho por bloques para evitar una matriz gigante en memoria."""
    pieces = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*stack.*")
        for chunk in pd.read_csv(path, chunksize=384, low_memory=False):
            # La primera columna puede representar el índice sin encabezado explícito.
            if "Timestamp" not in chunk:
                chunk = chunk.rename(columns={chunk.columns[0]: "Timestamp"})
            chunk["Timestamp"] = pd.to_datetime(chunk["Timestamp"], utc=True, errors="coerce")
            s = chunk.set_index("Timestamp").stack(dropna=True)
            s.index.names = ["fecha", "stream_id"]
            if len(s):
                pieces.append(s.reset_index(name="espectadores"))
    if not pieces:
        raise ValueError("No hay observaciones utilizables")
    all_obs = pd.concat(pieces, ignore_index=True)
    all_obs["espectadores"] = pd.to_numeric(all_obs["espectadores"], errors="coerce")
    all_obs = all_obs.dropna(subset=["fecha", "espectadores"])
    all_obs = all_obs[(all_obs["espectadores"] >= 0) &
                      np.isfinite(all_obs["espectadores"])]
    all_obs = (all_obs.groupby(["stream_id", "fecha"], as_index=False)
               ["espectadores"].mean().sort_values(["stream_id", "fecha"]))
    return all_obs.reset_index(drop=True)


def select_windows(obs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """Ventanas de ocho puntos; excluye discontinuidades y sesiones cortas."""
    by_id = {}
    for stream, g in obs.groupby("stream_id", sort=True):
        g = g.sort_values("fecha")
        secs = g["fecha"].diff().dt.total_seconds().fillna(300)
        segment_id = ((secs < 120) | (secs > 540)).cumsum()
        windows = []
        for _, segment in g.groupby(segment_id):
            values = segment["espectadores"].to_numpy(dtype=float)
            for start in range(0, len(values) - WINDOW + 1, WINDOW):
                win = values[start:start + WINDOW]
                if np.isfinite(win).all() and np.max(win) >= 10:
                    windows.append(win)
        if len(windows) >= 2:
            by_id[str(stream)] = np.asarray(windows, dtype=float)
    ids = sorted(by_id)
    if len(ids) < 40:
        raise RuntimeError(f"Solo {len(ids)} transmisiones elegibles; verificar muestreo")
    rng = np.random.default_rng(SEED)
    chosen = sorted(rng.choice(ids, size=min(TARGET_STREAMS, len(ids)), replace=False))
    split = int(len(chosen) * TRAIN_SHARE)
    train_ids = set(rng.permutation(chosen)[:split])
    train, test = [], []
    for sid in chosen:
        (train if sid in train_ids else test).extend(by_id[sid])
    if not train or not test:
        raise RuntimeError("La selección no produjo datos de entrenamiento y prueba")
    detail = {
        "observaciones_originales_agrupadas": len(obs),
        "streams_en_dataset": int(obs["stream_id"].nunique()),
        "streams_elegibles": len(ids),
        "streams_muestreados": len(chosen),
        "streams_train": len(train_ids),
        "streams_test": len(chosen) - len(train_ids),
        "ventanas_train": len(train),
        "ventanas_test": len(test),
        "mediciones_por_ventana": WINDOW,
        "criterio_continuidad_min_seg": 120,
        "criterio_continuidad_max_seg": 540,
    }
    return np.asarray(train), np.asarray(test), detail


def features(windows: np.ndarray) -> np.ndarray:
    x = np.asarray(windows, dtype=float)
    median = np.median(x, axis=1)
    scale = median + 20.0
    diff = np.diff(x, axis=1)
    first = np.mean(x[:, :3], axis=1)
    last = np.mean(x[:, -3:], axis=1)
    return np.column_stack((
        np.log1p(median),
        np.max(diff, axis=1) / scale,
        np.min(diff, axis=1) / scale,
        (x.max(axis=1) - x.min(axis=1)) / scale,
        np.std(x, axis=1) / scale,
        (last - first) / scale,
        (x[:, -1] - x[:, 0]) / scale,
    ))


def inject_attacks(real: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Escenarios de laboratorio; no describen ni etiquetan bots verdaderos."""
    rng = np.random.default_rng(SEED + 17)
    new = real.copy()
    types = np.array(["escalon", "pulso", "rampa"], dtype=object)
    label = types[np.arange(len(real)) % len(types)]
    for i, (x, kind) in enumerate(zip(new, label)):
        magnitude = max(float(np.median(real[i])), 30.) * rng.uniform(.8, 2.2)
        if kind == "escalon":
            x[4:] += magnitude
        elif kind == "pulso":
            x[3:6] += magnitude
        else:
            x[2:] += magnitude * np.array([.20, .40, .60, .80, 1., 1.])
    return new, label


def report(tr: np.ndarray, real: np.ndarray, injected: np.ndarray,
           names: np.ndarray) -> tuple[pd.DataFrame, dict, dict]:
    train = features(tr)
    fr, fs = features(real), features(injected)
    model = IsolationForest(n_estimators=150, max_samples=min(256, len(tr)),
                            random_state=SEED, n_jobs=1, contamination="auto")
    model.fit(train)
    scores = {
        "Umbral por salto": (train[:, 1], fr[:, 1], fs[:, 1]),
        "Isolation Forest": (-model.score_samples(train),
                            -model.score_samples(fr),
                            -model.score_samples(fs)),
    }
    rows, predictions, saved_scores = [], {}, {}
    for method, (sc_tr, sc_real, sc_sim) in scores.items():
        cutoff = float(np.quantile(sc_tr, QUANTILE))
        alert_real = sc_real > cutoff
        alert_sim = sc_sim > cutoff
        predictions[method] = (alert_real, alert_sim)
        saved_scores[method] = (sc_real, sc_sim)
        rows.append({"metodo": method, "umbral": cutoff,
                     "ventanas_reales_sin_etiqueta": len(real),
                     "alertas_reales": int(alert_real.sum()),
                     "tasa_alertas_reales": float(alert_real.mean()),
                     "escenarios_simulados": len(injected),
                     "detecciones_simuladas": int(alert_sim.sum()),
                     "recall_escenarios_simulados": float(alert_sim.mean()),
                     "aumento_alertas_pares": int(np.sum(alert_sim & ~alert_real))})
        for kind in ("escalon", "pulso", "rampa"):
            take = names == kind
            rows.append({"metodo": method, "escenario":kind,
                         "escenarios_simulados":int(take.sum()),
                         "detecciones_simuladas":int(alert_sim[take].sum()),
                         "recall_escenarios_simulados":float(alert_sim[take].mean())})
    return pd.DataFrame(rows), predictions, saved_scores


def plots(real: np.ndarray, simulated: np.ndarray, types: np.ndarray,
          metrics: pd.DataFrame, scores: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    # Primer caso escalón por índice predefinido (sin optimizar la elección).
    i = int(np.flatnonzero(types == "escalon")[0])
    fig, ax = plt.subplots(figsize=(9, 4))
    t = np.arange(WINDOW) * 5
    ax.plot(t, real[i], marker="o", label="Audiencia real observada")
    ax.plot(t, simulated[i], marker="s", label="Misma ventana + escalón simulado")
    ax.set(xlabel="Minutos desde el comienzo de la ventana",
           ylabel="Espectadores concurrentes",
           title="Ejemplo de alteración simulada (NO es evidencia de bots)")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(FIG/"ejemplo_escenario_simulado.png",dpi=175)
    plt.close(fig)

    overall = metrics[metrics["escenario"].isna()] if "escenario" in metrics else metrics
    fig, ax = plt.subplots(figsize=(8, 4))
    groups = np.arange(len(overall))
    b1=ax.bar(groups-.19,overall["tasa_alertas_reales"]*100,.38,label="Alertas sobre datos reales (sin verdad terreno)")
    b2=ax.bar(groups+.19,overall["recall_escenarios_simulados"]*100,.38,label="Detección de perturbaciones sintéticas")
    ax.bar_label(b1,fmt="%.1f",padding=2,fontsize=9)
    ax.bar_label(b2,fmt="%.1f",padding=2,fontsize=9)
    ax.set_xticks(groups,overall["metodo"].tolist())
    ax.set_ylim(0,105)
    ax.set_ylabel("Porcentaje de ventanas (%)")
    ax.set_title("Comparación exploratoria: NO equivale a detectar bots reales")
    ax.legend(fontsize=8,loc="upper left")
    ax.grid(axis="y",alpha=.2)
    fig.tight_layout()
    fig.savefig(FIG/"comparacion_metodos.png",dpi=175)
    plt.close(fig)

    sub=metrics[metrics["escenario"].notna()]
    fig,ax=plt.subplots(figsize=(8,4))
    scenarios=["escalon","pulso","rampa"]
    xx=np.arange(3)
    for j,method in enumerate(["Umbral por salto","Isolation Forest"]):
        values=[float(sub[(sub["metodo"]==method)&(sub["escenario"]==k)]["recall_escenarios_simulados"].iloc[0])*100 for k in scenarios]
        ax.bar(xx+(.18 if j else -.18),values,.36,label=method)
    ax.set_xticks(xx,["Escalón","Pulso","Rampa"])
    ax.set_ylim(0,105)
    ax.set_ylabel("Detección de perturbaciones sintéticas (%)")
    ax.set_title("Sensibilidad según el tipo de escenario simulado")
    ax.legend()
    ax.grid(axis="y",alpha=.2)
    fig.tight_layout()
    fig.savefig(FIG/"escenarios_simulados.png",dpi=175)
    plt.close(fig)


def main() -> None:
    path = download_source()
    obs = read_observations(path)
    train,test,detail = select_windows(obs)
    injected,kinds = inject_attacks(test)
    table,pred,scores=report(train,test,injected,kinds)
    plots(test,injected,kinds,table,scores)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    summary = table[table["escenario"].isna()].drop(columns=["escenario"])
    cases = table[table["escenario"].notna()][["metodo", "escenario",
        "escenarios_simulados", "detecciones_simuladas", "recall_escenarios_simulados"]]
    summary.to_csv(OUTPUT/"metricas.csv",index=False,float_format="%.6f")
    cases.to_csv(OUTPUT/"escenarios.csv",index=False,float_format="%.6f")
    detail.update({"dataset":"YTLive, Mozhganfar et al. (2026)",
                   "origen_csv":SOURCE,
                   "sha256_csv":hashlib.sha256(path.read_bytes()).hexdigest(),
                   "etiquetas_bots_reales":False,
                   "tipo_experimento":"Perturbaciones sintéticas sobre ventanas de audiencia real",
                   "semilla":SEED,"percentil_umbral_train":95,
                   "nota":"La tasa de alertas en datos reales NO es tasa de falsos positivos: no hay etiquetas de bots."})
    (OUTPUT/"resumen.json").write_text(json.dumps(detail,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(detail,ensure_ascii=False,indent=2))
    print("\nRESULTADOS (SIN EVIDENCIA DE BOTS REALES)")
    print(table.fillna("-").to_string(index=False))
    print("\nGráficos y métricas escritos en graficos/ y resultados/.")


if __name__=="__main__":
    main()
