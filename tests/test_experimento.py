"""Pruebas locales del diseño experimental.

Ejecutar: python -m unittest discover -s tests -v
No necesitan descargar el archivo YTLive ni tener inicializado el submódulo.
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from main import (EXPECTED_SHA256, FEATURE_NAMES, INT_COLUMNS, METHODS,
                  PERTURB_FLOOR, PERTURB_MAX, PERTURB_MIN, PERTURB_SEED, SEED,
                  WINDOW, download_source, features, file_sha256,
                  inject_attacks, read_observations, report, score_methods,
                  threshold, window_table)


class ExperimentoTests(unittest.TestCase):
    def setUp(self):
        self.real = np.array([
            [100, 105, 103, 107, 110, 108, 112, 111],
            [220, 215, 218, 217, 216, 214, 213, 216],
            [60, 60, 61, 61, 60, 61, 61, 62],
        ], dtype=float)
        self.train = np.tile(self.real, (25, 1))

    def test_ventanas_y_variables_sin_etiquetas(self):
        feat = features(self.real)
        self.assertEqual(feat.shape, (3, 7))
        self.assertEqual(feat.shape[1], len(FEATURE_NAMES))
        self.assertTrue(np.isfinite(feat).all())

    def test_simulaciones_son_deterministas_y_no_mutan_original(self):
        old = self.real.copy()
        altered, kinds = inject_attacks(self.real)
        altered_again, kinds_again = inject_attacks(self.real)
        np.testing.assert_array_equal(self.real, old)
        np.testing.assert_array_equal(altered, altered_again)
        np.testing.assert_array_equal(kinds, kinds_again)
        self.assertEqual(set(kinds), {"escalon", "pulso", "rampa"})
        self.assertTrue(np.all(altered >= self.real))

    def test_la_rampa_crece_de_manera_gradual(self):
        altered, kinds = inject_attacks(self.real)
        pos = int(np.flatnonzero(kinds == "rampa")[0])
        change = altered[pos] - self.real[pos]
        self.assertTrue(np.allclose(change[:2], 0))
        self.assertTrue(np.all(np.diff(change[2:]) >= 0))

    def test_parametros_de_perturbacion_declarados_en_el_resumen(self):
        # Los mismos valores que main() escribe en resultados/resumen.json.
        self.assertEqual(PERTURB_SEED, SEED + 17)
        self.assertEqual(PERTURB_SEED, 59)
        self.assertEqual((PERTURB_MIN, PERTURB_MAX), (.80, 2.20))
        self.assertEqual(PERTURB_FLOOR, 30.)

    def test_metricas_limites_y_datos_reales_sin_etiqueta(self):
        altered, kinds = inject_attacks(self.real)
        m = report(self.train, self.real, altered, kinds)
        summary = m[m["escenario"].isna()]
        self.assertEqual(len(summary), 2)
        self.assertTrue(summary["recall_escenarios_simulados"].between(0, 1).all())
        self.assertTrue(summary["tasa_alertas_reales"].between(0, 1).all())
        self.assertTrue((summary["ventanas_reales_sin_etiqueta"] == len(self.real)).all())
        self.assertTrue((summary["escenarios_simulados"] == len(self.real)).all())

    def test_los_conteos_se_escriben_como_enteros(self):
        altered, kinds = inject_attacks(self.real)
        m = report(self.train, self.real, altered, kinds)
        summary = m[m["escenario"].isna()].drop(columns=["escenario"])
        summary = summary.astype({c: "int64" for c in INT_COLUMNS})
        for column in INT_COLUMNS:
            self.assertEqual(summary[column].dtype, np.int64)
        cabecera, fila = summary.to_csv(index=False,
                                        float_format="%.6f").splitlines()[:2]
        escrito = dict(zip(cabecera.split(","), fila.split(",")))
        for column in INT_COLUMNS:
            self.assertRegex(escrito[column], r"^\d+$")
        # Las proporciones sí conservan seis decimales.
        self.assertRegex(escrito["tasa_alertas_reales"], r"^\d\.\d{6}$")

    def test_reutilizar_puntajes_no_cambia_las_metricas(self):
        altered, kinds = inject_attacks(self.real)
        scores = score_methods(self.train, self.real, altered)
        sin_cache = report(self.train, self.real, altered, kinds)
        con_cache = report(self.train, self.real, altered, kinds, scores)
        self.assertTrue(sin_cache.equals(con_cache))

    def test_detalle_por_ventana_sin_serie_de_espectadores(self):
        altered, kinds = inject_attacks(self.real)
        scores = score_methods(self.train, self.real, altered)
        detail = window_table(self.real, kinds, scores)
        self.assertEqual(len(detail), len(self.real))
        esperadas = ["ventana", *FEATURE_NAMES, "escenario"]
        for short in ("salto", "isolation"):
            esperadas += [f"puntaje_{short}", f"alerta_{short}",
                          f"puntaje_{short}_perturbado", f"alerta_{short}_perturbado"]
        self.assertEqual(sorted(detail.columns), sorted(esperadas))
        # Ninguna columna reproduce las mediciones originales de YTLive.
        self.assertEqual(len(detail.columns), 2 + len(FEATURE_NAMES) + 8)
        for valores in self.real:
            self.assertFalse(np.isin(valores, detail.select_dtypes("number").to_numpy()).all())

    def test_las_alertas_del_detalle_respetan_el_umbral(self):
        altered, kinds = inject_attacks(self.real)
        scores = score_methods(self.train, self.real, altered)
        detail = window_table(self.real, kinds, scores)
        for method, short in zip(METHODS, ("salto", "isolation")):
            sc_tr, sc_real, sc_sim = scores[method]
            cutoff = threshold(sc_tr)
            np.testing.assert_array_equal(detail[f"alerta_{short}"],
                                          sc_real > cutoff)
            np.testing.assert_array_equal(detail[f"alerta_{short}_perturbado"],
                                          sc_sim > cutoff)

    def test_el_detalle_coincide_con_las_metricas_agregadas(self):
        altered, kinds = inject_attacks(self.real)
        scores = score_methods(self.train, self.real, altered)
        detail = window_table(self.real, kinds, scores)
        m = report(self.train, self.real, altered, kinds, scores)
        summary = m[m["escenario"].isna()].set_index("metodo")
        for method, short in zip(METHODS, ("salto", "isolation")):
            fila = summary.loc[method]
            self.assertEqual(int(detail[f"alerta_{short}"].sum()),
                             int(fila["alertas_reales"]))
            self.assertEqual(int(detail[f"alerta_{short}_perturbado"].sum()),
                             int(fila["detecciones_simuladas"]))
            nuevas = detail[f"alerta_{short}_perturbado"] & ~detail[f"alerta_{short}"]
            self.assertEqual(int(nuevas.sum()), int(fila["aumento_alertas_pares"]))

    def test_lectura_de_csv_ancho_sin_tratar_vacios_como_cero(self):
        # Reproduce el formato de YTLive: una columna por transmisión y celdas
        # vacías. Cubre la compatibilidad de stack() entre pandas 2.x y 3.x.
        contenido = ("Timestamp,aaa,bbb\n"
                     "2024-01-01T00:00:00Z,10,20\n"
                     "2024-01-01T00:05:00Z,11,\n"
                     "2024-01-01T00:10:00Z,12,21\n")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "viewers.csv"
            path.write_text(contenido, encoding="utf-8")
            obs = read_observations(path)
        self.assertEqual(sorted(obs.columns), ["espectadores", "fecha", "stream_id"])
        self.assertEqual(len(obs), 5)          # la celda vacía no cuenta como 0
        self.assertEqual(set(obs["stream_id"]), {"aaa", "bbb"})
        self.assertEqual(obs[obs["stream_id"] == "bbb"]["espectadores"].tolist(),
                         [20., 21.])

    def test_usa_el_submodulo_local_si_el_hash_coincide(self):
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "ytlive" / "viewers.csv"
            local.parent.mkdir(parents=True)
            local.write_bytes(b"Timestamp,aaa\n2024-01-01T00:00:00Z,10\n")
            with mock.patch.multiple("main", CACHE=Path(tmp) / "origen",
                                     SUBMODULE_CSV=local,
                                     EXPECTED_SHA256=file_sha256(local)):
                self.assertEqual(download_source(), local)

    def test_ignora_el_submodulo_si_el_hash_no_coincide(self):
        # Una copia alterada nunca debe usarse en silencio: se intenta descargar.
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "ytlive" / "viewers.csv"
            local.parent.mkdir(parents=True)
            local.write_bytes(b"datos modificados\n")
            with mock.patch.multiple("main", CACHE=Path(tmp) / "origen",
                                     SUBMODULE_CSV=local), \
                 mock.patch("main.urlopen", side_effect=OSError("sin red")):
                with self.assertRaises(OSError):
                    download_source()

    def test_sha256_tiene_formato_esperado(self):
        self.assertEqual(len(EXPECTED_SHA256), 64)
        self.assertTrue(all(x in "0123456789abcdef" for x in EXPECTED_SHA256))

    def test_ventana_de_ocho_mediciones(self):
        self.assertEqual(WINDOW, self.real.shape[1])


if __name__ == "__main__":
    unittest.main()
