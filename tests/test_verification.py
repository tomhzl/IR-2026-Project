import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.verification import answer_type_error, validate_audit, focus_snippets, can_adjudicate, audit_candidate, source_span


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.snippets = [{'id': 'S1', 'doc_id': 'd1', 'text':
                          'Alpha University founded the Orion project in 1984. The founder was a university.'}]
        self.audit = {'supported': True, 'type_matches': True, 'missing_clues': [],
                      'answer_support': [{'snippet_id': 'S1', 'quote': 'Alpha University founded the Orion project in 1984.'}],
                      'checks': [{'claim': 'Orion was founded by Alpha University', 'status': 'supported',
                                  'snippet_id': 'S1', 'quote': 'Alpha University founded the Orion project in 1984.'}]}

    def test_valid_source_grounded_audit(self):
        supported, _, ids = validate_audit(self.audit, self.snippets, 'Who founded Orion?')
        self.assertTrue(supported)
        self.assertEqual(ids, ['S1'])

    def test_typographic_alignment_returns_original_span(self):
        text = '11-year-old Boyko wants a "brazuca" football.'
        aligned = source_span('11 year old Boyko wants a brazuca', text)
        self.assertIsNotNone(aligned)
        self.assertIn(aligned, text)
        self.assertIn('"brazuca', aligned)
        self.assertEqual(source_span('The value was 7%', 'The value was 7 %.'), 'The value was 7 %')

    def test_alignment_cannot_change_numbers_signs_or_skip_words(self):
        self.assertIsNone(source_span('The value was 14.81%', 'The value was 0.818.'))
        self.assertIsNone(source_span('The value was +5', 'The value was -5.'))
        self.assertIsNone(source_span('He founded Orion', 'He never founded Orion.'))
        self.assertIsNone(source_span('The price is $5', 'The price is €5.'))
        self.assertIsNone(source_span('The value was 5', 'The value was 50.'))

    def test_explicit_ellipsis_is_expanded_to_the_original_continuous_text(self):
        text = 'Reviews criticized repetitive combat. Other details are discussed. It is a decent game.'
        span = source_span('Reviews criticized repetitive combat... It is a decent game.', text)
        self.assertIn('Other details are discussed', span)
        self.assertIn(span, text)

    def test_positive_verdict_can_extract_an_omitted_literal_entity_anchor(self):
        audit = copy.deepcopy(self.audit)
        audit['answer_support'] = []
        self.assertTrue(validate_audit(audit, self.snippets, 'Who founded Orion?', 'Alpha University')[0])
        self.assertFalse(validate_audit(audit, self.snippets, 'Who founded Orion?', 'Beta University')[0])

    def test_supported_boolean_without_checks_is_rejected(self):
        self.assertFalse(validate_audit({'supported': True}, self.snippets, 'Who founded Orion?')[0])

    def test_short_direct_name_still_needs_grounded_relationship_checks(self):
        audit = copy.deepcopy(self.audit)
        audit['answer_support'][0]['quote'] = 'Orion'
        audit['unverified_background'] = ['An incidental historical detail is not in the excerpts.']
        self.assertTrue(validate_audit(audit, self.snippets, 'Name the project.')[0])
        audit['checks'] = []
        self.assertFalse(validate_audit(audit, self.snippets, 'Name the project.')[0])

    def test_invented_quote_and_unknown_id_are_rejected(self):
        for key, value in [('quote', 'Beta University founded the Orion project in 1984.'), ('snippet_id', 'S99')]:
            audit = copy.deepcopy(self.audit)
            audit['answer_support'][0][key] = value
            self.assertFalse(validate_audit(audit, self.snippets, 'Who founded Orion?')[0])

    def test_cannot_support_one_candidate_with_another_candidates_quote(self):
        self.assertFalse(validate_audit(self.audit, self.snippets, 'Who founded Orion?', 'Beta University')[0])
        self.assertTrue(validate_audit(self.audit, self.snippets, 'Who founded Orion?', 'Alpha University')[0])

    def test_focused_context_preserves_same_document_pronoun_evidence(self):
        snippets = [{'id': 'S1', 'doc_id': 'person', 'text': 'Alice Example performed stage magic.'},
                    {'id': 'S2', 'doc_id': 'person', 'text': 'She drove a car while blindfolded in 1987.'}]
        snippets.extend({'id': f'X{i}', 'doc_id': f'noise{i}', 'text': 'An unrelated historical document.'}
                        for i in range(16))
        selected = focus_snippets('Who performed magic and drove while blindfolded?', 'Alice Example', snippets)
        self.assertIn('S2', [item['id'] for item in selected])
        self.assertLessEqual(len(selected), 14)

    def test_missing_clue_or_wrong_type_is_rejected(self):
        for key, value in [('missing_clues', ['The date was not established']), ('type_matches', False)]:
            audit = copy.deepcopy(self.audit)
            audit[key] = value
            self.assertFalse(validate_audit(audit, self.snippets, 'Who founded Orion?')[0])

    def test_long_unumbered_questions_need_distinct_checks(self):
        question = 'Identify the founder. ' + 'A background condition is described here. ' * 12
        self.assertFalse(validate_audit(self.audit, self.snippets, question)[0])
        audit = copy.deepcopy(self.audit)
        audit['checks'] *= 2
        self.assertFalse(validate_audit(audit, self.snippets, question)[0])
        audit['checks'][1] = {'claim': 'The project was founded in 1984', 'status': 'supported',
                              'snippet_id': 'S1', 'quote': 'Alpha University founded the Orion project in 1984.'}
        self.assertTrue(validate_audit(audit, self.snippets, question)[0])

    def test_person_and_name_questions_reject_numeric_answers(self):
        self.assertIsNotNone(answer_type_error('Who won the event two years before X did?', '2008'))
        self.assertIsNotNone(answer_type_error('What is the name of the ability?', '2008'))
        self.assertIsNone(answer_type_error('When did the event occur?', '2008'))
        self.assertIsNone(answer_type_error('Who won the event?', 'Svetlana Gromenkova'))

    def grounded_partial(self):
        audit = copy.deepcopy(self.audit)
        audit.update(supported=False, missing_clues=['An unresolved background detail.'])
        audit['checks'].append({'claim': 'The project was founded in 1984', 'status': 'supported',
                                'snippet_id': 'S1', 'quote': 'Alpha University founded the Orion project in 1984.'})
        return audit

    def test_adjudication_requires_two_distinct_grounded_checks(self):
        partial = self.grounded_partial()
        self.assertTrue(can_adjudicate(partial, self.snippets, 'Alpha University', 'Who founded Orion?'))
        partial['checks'][1]['quote'] = 'This is invented evidence.'
        self.assertFalse(can_adjudicate(partial, self.snippets, 'Alpha University', 'Who founded Orion?'))
        partial = self.grounded_partial()
        partial['checks'].append({'claim': 'A contradictory condition', 'status': 'contradicted'})
        self.assertFalse(can_adjudicate(partial, self.snippets, 'Alpha University', 'Who founded Orion?'))

    def test_adjudication_has_its_own_verdict_log_and_token_usage(self):
        partial = self.grounded_partial()
        final = {**partial, 'supported': True, 'missing_clues': []}
        generator = Mock(side_effect=[(partial, 20, 5), (final, 12, 7)])
        tools = Mock()
        supported, ids, inputs, outputs, error = audit_candidate(
            'Who founded Orion?', 'Alpha University', self.snippets, 'test-model', tools, generator)
        self.assertTrue(supported)
        self.assertEqual(ids, ['S1'])
        self.assertEqual((inputs, outputs), (32, 12))
        self.assertIsNone(error)
        self.assertIn('verification_hint', generator.call_args.kwargs)
        stages = [call.kwargs['stage'] for call in tools.record_retrieval.call_args_list]
        self.assertEqual(stages[-2:], ['answer_verification', 'answer_verification_adjudication'])


if __name__ == '__main__':
    unittest.main()
