from __future__ import annotations

import unittest

import pandas as pd

from pipelines.label.bge_teacher import add_teacher_labels


class BGEDistillationTests(unittest.TestCase):
    def test_add_teacher_labels_defaults_to_scores_only(self) -> None:
        rows = pd.DataFrame(
            {
                "query": ["running shoes"],
                "product_id": ["unjudged"],
                "label": [0.0],
                "split": ["train"],
                "bge_teacher_score": [0.95],
                "features": [{}],
            }
        )

        got = add_teacher_labels(rows, judged_pairs=set())

        self.assertEqual(got["label"].tolist(), [0.0])
        self.assertEqual(got["sample_weight"].tolist(), [1.0])
        self.assertEqual(got.loc[0, "features"]["bge_teacher_score"], 0.95)

    def test_add_teacher_labels_only_upgrades_train_unjudged_rows_with_lower_weight(self) -> None:
        rows = pd.DataFrame(
            {
                "query": ["running shoes", "running shoes", "running shoes", "running shoes"],
                "product_id": ["exact", "known_bad", "unjudged_train", "unjudged_val"],
                "label": [4.0, 0.0, 0.0, 0.0],
                "split": ["train", "train", "train", "val"],
                "bge_teacher_score": [0.10, 0.99, 0.95, 0.94],
                "features": [{}, {}, {}, {}],
            }
        )

        got = add_teacher_labels(
            rows,
            judged_pairs={("running shoes", "exact"), ("running shoes", "known_bad")},
            pseudo_labels_enabled=True,
            pseudo_weight=0.25,
        )

        labels = dict(zip(got["product_id"], got["label"]))
        self.assertEqual(labels["exact"], 4.0)
        self.assertEqual(labels["known_bad"], 0.0)
        self.assertEqual(labels["unjudged_train"], 2.0)
        self.assertEqual(labels["unjudged_val"], 0.0)
        self.assertEqual(
            got.loc[got["product_id"] == "unjudged_train", "sample_weight"].item(),
            0.25,
        )
        self.assertEqual(
            got.loc[got["product_id"] == "unjudged_val", "sample_weight"].item(),
            1.0,
        )
        self.assertEqual(
            got.loc[got["product_id"] == "unjudged_train", "features"].item()["bge_teacher_score"],
            0.95,
        )
        self.assertGreater(
            got.loc[got["product_id"] == "unjudged_train", "features"].item()["bge_teacher_percentile"],
            0.8,
        )

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
