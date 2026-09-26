"""Offline reversible-record contracts; synthetic text, no database or provider."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import unittest

from memory_demo.retrieval.recall_records import (
    RecordReferenceError, align_window, model_records, project_source,
    project_window, resolve_record,
)


TEXT = (
    "[source_key: sample.json]\n[segment_index: 0]\n\n"
    "[speaker_alias_legend]\n"
    "person-a: zh-CN=甲 | en=Alice | ko=가\n"
    "person-b: zh-CN=乙 | en=Bob\n\n"
    "[record: 10]\n[speaker_raw: person-a]\n[script_raw: 원문.]\n"
    "zh-CN: 甲说：[tooltip=解释]同意[/tooltip]。\nen: Alice agrees.\n\n"
    "[record: 11]\n[speaker_raw: person-b]\n[script_raw: 두 번째.]\n"
    "zh-CN: 乙说：同意。\nen: Bob agrees.\n"
)


class RecallRecordsTests(unittest.TestCase):
    def records(self, **kwargs):
        return project_source(7, TEXT, [8, 9], **kwargs)

    def test_chinese_default_is_an_exact_raw_span_and_markup_is_kept(self):
        records = self.records()
        self.assertEqual([r['record_index'] for r in records], [10, 11])
        self.assertEqual(records[0]['language'], 'zh-CN')
        self.assertEqual(records[0]['text'], '甲说：[tooltip=解释]同意[/tooltip]。')
        for record in records:
            evidence = resolve_record(record['record_id'], records, episode_id=8, raw=TEXT)
            self.assertEqual(evidence['quote'], TEXT[evidence['start']:evidence['end']])
            self.assertEqual(evidence['quote'], record['text'])
            self.assertEqual(record['raw_record'], TEXT[record['context_start']:record['context_end']])
            self.assertEqual(evidence['source_sha256'], hashlib.sha256(TEXT.encode()).hexdigest())

    def test_requested_language_changes_display_not_record_identity(self):
        zh, en = self.records(), self.records(language='en')
        self.assertEqual([r['record_id'] for r in zh], [r['record_id'] for r in en])
        self.assertEqual(en[0]['text'], 'Alice agrees.')
        self.assertEqual(en[0]['language'], 'en')
        self.assertEqual(TEXT[en[0]['start']:en[0]['end']], en[0]['text'])

    def test_missing_language_prefers_stored_original_then_available_language(self):
        fallback = self.records(language='ja')[0]
        self.assertEqual(fallback['language'], 'script_raw')
        self.assertEqual(fallback['text'], '원문.')
        raw = '[record: 1]\nen: First line.\nSecond line.\n'
        record = project_source(1, raw, [1])[0]
        self.assertEqual(record['language'], 'en')
        self.assertEqual(record['text'], 'First line.\nSecond line.')
        self.assertEqual(raw[record['start']:record['end']], record['text'])

    def test_existing_aliases_and_legend_keep_raw_provenance(self):
        record = self.records()[0]
        self.assertEqual(record['speaker_raw'], 'person-a')
        self.assertEqual(record['speaker_aliases'], {'zh-CN': '甲', 'en': 'Alice', 'ko': '가'})
        legend = record['speaker_alias_legend']
        self.assertEqual(legend['text'], TEXT[legend['start']:legend['end']])
        self.assertEqual(legend['aliases']['person-b']['en'], 'Bob')
        self.assertEqual(model_records([record])[0]['speaker_aliases'], record['speaker_aliases'])

    def test_inline_aliases_are_supported_without_inventing_unknown_speakers(self):
        raw = '[record: 1]\n[speaker_raw: a]\n[speaker_aliases: zh-CN=甲 | en=A]\nzh-CN: 到了。\n\n[record: 2]\nzh-CN: 谁到了？'
        first, second = project_source(1, raw, [1])
        self.assertEqual(first['speaker_aliases'], {'zh-CN': '甲', 'en': 'A'})
        self.assertIsNone(second['speaker_raw'])
        self.assertEqual(second['speaker_aliases'], {})

    def test_duplicate_sentences_in_different_records_have_distinct_references(self):
        raw = '[record: 1]\nzh-CN: 同意。\n\n[record: 2]\nzh-CN: 同意。'
        records = project_source(1, raw, [3])
        self.assertNotEqual(records[0]['record_id'], records[1]['record_id'])
        resolved = [resolve_record(r['record_id'], records, episode_id=3, raw=raw) for r in records]
        self.assertEqual([r['quote'] for r in resolved], ['同意。', '同意。'])
        self.assertNotEqual(resolved[0]['start'], resolved[1]['start'])

    def test_cut_tail_stays_unread_until_the_complete_record_is_visible(self):
        all_records = self.records()
        second = all_records[1]
        cut = second['start'] + 2
        partial = project_window(7, TEXT, [8], end=cut)
        self.assertEqual([r['record_index'] for r in partial.records], [10])
        self.assertEqual(partial.consumed_end, second['context_start'])
        self.assertEqual(partial.incomplete_ranges, [{'record_index': 11, 'start': second['context_start'], 'end': second['context_end'], 'missing_prefix': False, 'missing_suffix': True}])
        continued = project_window(7, TEXT, [8], start=partial.consumed_end)
        self.assertEqual([r['record_index'] for r in continued.records], [11])
        self.assertEqual(continued.consumed_end, len(TEXT))
        with self.assertRaises(RecordReferenceError):
            resolve_record(second['record_id'], partial.records, episode_id=8)

    def test_cut_prefix_reports_rewind_even_if_later_records_are_complete(self):
        first = self.records()[0]
        start = first['start'] + 1
        projection = project_window(7, TEXT, [8], start=start)
        self.assertEqual([r['record_index'] for r in projection.records], [11])
        self.assertTrue(projection.incomplete_ranges[0]['missing_prefix'])
        self.assertEqual(projection.consumed_end, first['context_start'])
        self.assertLess(projection.consumed_end, start)
        aligned = align_window(TEXT, start, len(TEXT))
        self.assertEqual(aligned, (first['context_start'], len(TEXT)))
        self.assertEqual(len(self.records(start=aligned[0], end=aligned[1])), 2)

    def test_empty_windows_do_not_claim_an_intersecting_record(self):
        inside = self.records()[0]['start'] + 1
        projection = project_window(7, TEXT, [8], start=inside, end=inside)
        self.assertEqual(projection.records, [])
        self.assertEqual(projection.incomplete_ranges, [])
        self.assertEqual(projection.consumed_end, inside)
        self.assertEqual(align_window(TEXT, inside, inside), (inside, inside))

    def test_plain_text_fallback_preserves_paragraphs_and_crlf(self):
        raw = 'First line.\r\nSecond [tooltip=x]line[/tooltip].\r\n\r\nNext paragraph.'
        records = project_source(1, raw, [2])
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]['text'], 'First line.\r\nSecond [tooltip=x]line[/tooltip].')
        self.assertTrue(all(r['language'] == 'raw' and r['record_index'] is None for r in records))
        for record in records:
            evidence = resolve_record(record['record_id'], records, episode_id=2, raw=raw)
            self.assertEqual(evidence['quote'], raw[evidence['start']:evidence['end']])
        self.assertEqual(project_window(1, raw, [2], end=5).records, [])
        self.assertEqual(project_window(1, raw, [2], end=5).consumed_end, 0)

    def test_formatted_crlf_and_multiline_language_are_reversible(self):
        raw = '[record: 2]\r\n[speaker_raw: a]\r\nzh-CN: 第一行。\r\n第二行。\r\nen: First.\r\nSecond.\r\n'
        record = project_source(1, raw, [1])[0]
        self.assertEqual(record['text'], '第一行。\r\n第二行。')
        self.assertEqual(raw[record['start']:record['end']], record['text'])

    def test_current_source_changes_or_old_id_in_new_projection_are_rejected(self):
        records = self.records()
        changed = TEXT.replace('Alice agrees.', 'Alice disagrees.')
        with self.assertRaisesRegex(RecordReferenceError, 'Source changed'):
            resolve_record(records[0]['record_id'], records, episode_id=8, raw=changed)
        with self.assertRaisesRegex(RecordReferenceError, 'not visible'):
            resolve_record(records[0]['record_id'], project_source(7, changed, [8]), episode_id=8)

    def test_unknown_id_wrong_episode_and_foreign_source_are_rejected(self):
        records = self.records()
        for record_id, episode in [('r_missing', 8), (records[0]['record_id'], 999), (records[0]['record_id'], True)]:
            with self.subTest(record_id=record_id, episode=episode), self.assertRaises(RecordReferenceError):
                resolve_record(record_id, records, episode_id=episode)
        other = project_source(6, TEXT, [8])
        with self.assertRaises(RecordReferenceError):
            resolve_record(other[0]['record_id'], records, episode_id=8)

    def test_explicit_span_is_absolute_contiguous_and_confined_to_displayed_text(self):
        records = self.records()
        record = records[0]
        start = TEXT.index('同意')
        evidence = resolve_record(record['record_id'], records, episode_id=8, start=start, end=start+2, raw=TEXT)
        self.assertEqual(evidence['quote'], '同意')
        for a, b in [(None, start), (start, None), (0, 2), (record['start'], record['end']+1), (start, start), (True, 2)]:
            with self.subTest(start=a, end=b), self.assertRaises(RecordReferenceError):
                resolve_record(record['record_id'], records, episode_id=8, start=a, end=b)

    def test_modified_projection_and_ambiguous_languages_are_rejected(self):
        records = self.records()
        damaged = deepcopy(records)
        damaged[0]['text'] = 'invented continuous quotation'
        with self.assertRaises(RecordReferenceError):
            resolve_record(records[0]['record_id'], damaged, episode_id=8)
        with self.assertRaisesRegex(RecordReferenceError, 'ambiguous'):
            resolve_record(records[0]['record_id'], records+self.records(language='en'), episode_id=8)
        duplicate = records+deepcopy(records)
        self.assertEqual(resolve_record(records[0]['record_id'], duplicate, episode_id=8)['quote'], records[0]['text'])

    def test_compact_view_and_resolution_do_not_mutate_local_credentials(self):
        records = self.records()
        before = deepcopy(records)
        compact = model_records(records)
        self.assertNotIn('raw_record', compact[0])
        self.assertNotIn('source_sha256', compact[0])
        compact[0]['speaker_aliases']['en'] = 'changed in provider input'
        resolve_record(records[0]['record_id'], records, episode_id=9, raw=TEXT)
        self.assertEqual(records, before)

    def test_invalid_projection_inputs_and_empty_sources(self):
        self.assertEqual(project_source(1, '', [1]), [])
        for source, episodes, start, end in [(True, [1], 0, 1), (1, [], 0, 1), (1, [False], 0, 1), (1, [1], -1, 1), (1, [1], 5, 2)]:
            with self.subTest(source=source, episodes=episodes, start=start, end=end), self.assertRaises(RecordReferenceError):
                project_source(source, TEXT, episodes, start=start, end=end)


if __name__ == '__main__':
    unittest.main()
