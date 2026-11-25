import unittest
from unittest.mock import patch

import run_full


class FullRunTests(unittest.TestCase):
    @patch('run_full.run_step', return_value=0)
    def test_dev_runs_summary_check_and_evaluation(self, step):
        self.assertEqual(run_full.main(['--split', 'dev', '--output', 'output/test-dev']), 0)
        names = [call.args[0][1].name for call in step.call_args_list]
        self.assertEqual(names, ['run_qwen_hybrid.py', 'summarize_run.py',
                                 'check_results.py', 'evaluate.py'])

    @patch('run_full.run_step', return_value=0)
    def test_test_split_does_not_require_private_labels(self, step):
        self.assertEqual(run_full.main(['--split', 'test', '--output', 'output/test-test']), 0)
        names = [call.args[0][1].name for call in step.call_args_list]
        self.assertEqual(names, ['run_qwen_hybrid.py', 'summarize_run.py', 'check_results.py'])

    @patch('run_full.run_step', return_value=2)
    def test_failed_step_stops_the_workflow(self, step):
        self.assertEqual(run_full.main(['--split', 'dev']), 2)
        self.assertEqual(step.call_count, 1)


if __name__ == '__main__':
    unittest.main()
