import unittest

from experiments.train_lora import tokenize_row


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize,
                            add_generation_prompt):
        if add_generation_prompt:
            return [10, 11, 12]
        return [10, 11, 12, 20, 21, 2]


class TokenizeRowTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            'id': '1',
            'split': 'train',
            'messages': [
                {'role': 'system', 'content': 'system'},
                {'role': 'user', 'content': 'question'},
                {'role': 'assistant', 'content': '{"answer":""}'},
            ],
        }

    def test_only_assistant_tokens_are_supervised(self):
        row = tokenize_row(self.row, FakeTokenizer(), 32)
        self.assertEqual(row['input_ids'], [10, 11, 12, 20, 21, 2])
        self.assertEqual(row['labels'], [-100, -100, -100, 20, 21, 2])
        self.assertEqual(row['attention_mask'], [1] * 6)

    def test_overlength_sample_is_rejected_not_truncated(self):
        with self.assertRaisesRegex(ValueError, 'exceed'):
            tokenize_row(self.row, FakeTokenizer(), 5)


if __name__ == '__main__':
    unittest.main()
