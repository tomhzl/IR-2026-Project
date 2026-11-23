import unittest

from summarize_run import summarize


class RunSummaryTests(unittest.TestCase):
    def info(self):
        return {'release': 'IR2026-BCP-200-v1', 'protocol_version': 'ir2026.retrieval.v2',
                'method': 'test-method', 'split': 'dev', 'execution_status': 'complete', 'queries': 2}

    def row(self, status, elapsed, input_tokens, output_tokens, complete=True):
        return {'status': status, 'usage': {
            'elapsed_seconds': elapsed, 'model_input_tokens': input_tokens,
            'model_output_tokens': output_tokens, 'tool_calls_complete': complete}}

    def test_complete_usage_is_aggregated(self):
        summary = summarize([
            self.row('ok', 1.5, 10, 1), self.row('no_answer', 2.5, 20, 3)], self.info())
        self.assertEqual(summary['queries'], 2)
        self.assertEqual(summary['status_counts']['ok'], 1)
        self.assertEqual(summary['status_counts']['no_answer'], 1)
        self.assertEqual(summary['completion_rate'], 0.5)
        self.assertEqual(summary['elapsed_seconds_total'], 4.0)
        self.assertEqual(summary['elapsed_seconds_mean'], 2.0)
        self.assertEqual(summary['elapsed_seconds_max'], 2.5)
        self.assertEqual(summary['model_input_tokens'], 30)
        self.assertEqual(summary['model_output_tokens'], 4)
        self.assertEqual(summary['token_complete_queries'], 2)
        self.assertEqual(summary['token_incomplete_queries'], 0)

    def test_missing_tokens_make_totals_null_not_zero(self):
        summary = summarize([
            self.row('ok', 1.0, 10, 1), self.row('error', 2.0, None, None, complete=False)], self.info())
        self.assertIsNone(summary['model_input_tokens'])
        self.assertIsNone(summary['model_output_tokens'])
        self.assertEqual(summary['token_complete_queries'], 1)
        self.assertEqual(summary['token_incomplete_queries'], 1)
        self.assertEqual(summary['incomplete_tool_usage_queries'], 1)
        self.assertIn('null', summary['token_note'])


if __name__ == '__main__':
    unittest.main()
