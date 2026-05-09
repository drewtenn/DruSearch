from __future__ import annotations

import unittest

import pandas as pd

from pipelines.label.bge_teacher import add_teacher_labels


class BGEDistillationTests(unittest.TestCase):
    def test_add_teacher_labels_only_upgrades_unjudged_rows(self) -> None:
        rows = pd.DataFrame(
            {
                "query": ["running shoes", "running shoes", "running shoes"],
                "product_id": ["exact", "known_bad", "unjudged"],
                "label": [4.0, 0.0, 0.0],
                "bge_teacher_score": [0.10, 0.99, 0.95],
                "features": [{}, {}, {}],
            }
        )

        got = add_teacher_labels(rows, judged_pairs={("running shoes", "exact"), ("running shoes", "known_bad")})

        labels = dict(zip(got["product_id"], got["label"]))
        self.assertEqual(labels["exact"], 4.0)
        self.assertEqual(labels["known_bad"], 0.0)
        self.assertEqual(labels["unjudged"], 2.0)
        self.assertEqual(got.loc[got["product_id"] == "unjudged", "features"].item()["bge_teacher_score"], 0.95)
        self.assertGreater(got.loc[got["product_id"] == "unjudged", "features"].item()["bge_teacher_percentile"], 0.8)

    def test_add_teacher_labels_does_not_create_signal_for_flat_teacher_scores(self) -> None:
        rows = pd.DataFrame(
            {
                "query": ["hats", "hats"],
                "product_id": ["a", "b"],
                "label": [0.0, 0.0],
                "bge_teacher_score": [0.42, 0.42],
                "features": [{}, {}],
            }
        )

        got = add_teacher_labels(rows, judged_pairs=set())

        self.assertEqual(got["label"].tolist(), [0.0, 0.0])
        self.assertEqual(
            got["features"].tolist(),
            [
                {"bge_teacher_score": 0.42, "bge_teacher_percentile": 0.0},
                {"bge_teacher_score": 0.42, "bge_teacher_percentile": 0.0},
            ],
        )


if __name__ == "__main__":
    unittest.main()
