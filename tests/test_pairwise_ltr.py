import unittest

from experiments.train_pairwise_ltr import (
    cross_validate,
    evaluate,
    fit_model,
    paired_rows,
    pairwise_data,
    rank_row,
)


def synthetic_row(qid, signal=1.0):
    return {
        'id': qid,
        'baseline_doc_ids': ['bad', 'good'],
        'candidates': [
            {
                'doc_id': 'bad',
                'features': {'signal': 0.0, 'reverse': 1.0},
            },
            {
                'doc_id': 'good',
                'features': {'signal': signal, 'reverse': 0.0},
            },
        ],
    }


class PairwiseLTRTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            synthetic_row(f'q{i}') for i in range(6)
        ]
        self.labels = {
            row['id']: {
                'gold_doc_ids': ['good'],
                'evidence_doc_ids': ['good'],
            }
            for row in self.rows
        }

    def test_pairwise_data_is_symmetric_and_query_grounded(self):
        x, y, queries, pairs = pairwise_data(
            self.rows[:2], self.labels, ['signal']
        )
        self.assertEqual((queries, pairs), (2, 2))
        self.assertEqual(x.shape, (4, 1))
        self.assertEqual(y.tolist(), [1, 1, 0, 0])
        self.assertEqual(
            x[:, 0].tolist(), [1.0, 1.0, -1.0, -1.0]
        )

    def test_fit_and_cross_validation_rank_whole_queries(self):
        model = fit_model(
            self.rows, self.labels, ['signal'], c=1.0
        )
        self.assertEqual(
            rank_row(self.rows[0], model), ['good', 'bad']
        )

        settings = {
            'folds': 3,
            'max_iter': 2000,
            'random_state': 2026,
        }
        metrics, folds, oof = cross_validate(
            self.rows,
            self.labels,
            ['signal'],
            1.0,
            settings,
        )
        self.assertEqual(
            set(oof), {row['id'] for row in self.rows}
        )
        self.assertEqual(len(folds), 3)
        self.assertEqual(metrics['gold_hit_at_10'], 1.0)
        self.assertEqual(
            evaluate(
                self.rows, self.labels, oof
            )['gold_hit_at_1'],
            1.0,
        )

    def test_paired_outcomes_do_not_hide_losses(self):
        baseline = {
            'q0': ['good'],
            'q1': ['good'],
            'q2': ['bad'],
        }
        proposed = {
            'q0': ['good'],
            'q1': ['bad'],
            'q2': ['good'],
        }
        rows = paired_rows(
            self.rows[:3],
            self.labels,
            baseline,
            proposed,
        )
        self.assertEqual(
            [row['outcome'] for row in rows],
            ['tie', 'loss', 'win'],
        )


if __name__ == '__main__':
    unittest.main()
