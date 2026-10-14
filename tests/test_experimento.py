"""Pruebas locales del diseño experimental.

Ejecutar: python -m unittest discover -s tests -v
No necesitan descargar el archivo YTLive.
"""
import unittest

import numpy as np

from main import EXPECTED_SHA256, features, inject_attacks, report, WINDOW


class ExperimentoTests(unittest.TestCase):
    def setUp(self):
        self.real = np.array([
            [100, 105, 103, 107, 110, 108, 112, 111],
            [220, 215, 218, 217, 216, 214, 213, 216],
            [60, 60, 61, 61, 60, 61, 61, 62],
        ], dtype=float)

    def test_ventanas_y_variables_sin_etiquetas(self):
        feat = features(self.real)
        self.assertEqual(feat.shape, (3, 7))
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

    def test_metricas_limites_y_datos_reales_sin_etiqueta(self):
        train = np.tile(self.real, (25, 1))
        altered, kinds = inject_attacks(self.real)
        m = report(train, self.real, altered, kinds)
        summary = m[m["escenario"].isna()]
        self.assertEqual(len(summary), 2)
        self.assertTrue(summary["recall_escenarios_simulados"].between(0, 1).all())
        self.assertTrue(summary["tasa_alertas_reales"].between(0, 1).all())
        self.assertTrue((summary["ventanas_reales_sin_etiqueta"] == len(self.real)).all())
        self.assertTrue((summary["escenarios_simulados"] == len(self.real)).all())

    def test_sha256_tiene_formato_esperado(self):
        self.assertEqual(len(EXPECTED_SHA256), 64)
        self.assertTrue(all(x in "0123456789abcdef" for x in EXPECTED_SHA256))


if __name__ == "__main__":
    unittest.main()
