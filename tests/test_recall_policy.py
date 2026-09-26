import unittest

from memory_demo.cli import build_parser
from memory_demo.retrieval.recall_policy import RecallPolicy, requests_maximum_recall


class RecallPolicyTests(unittest.TestCase):
    def test_modes_are_request_local_and_bounded(self):
        self.assertEqual(RecallPolicy.for_request("问题", "deep").timeout_seconds, 360)
        self.assertEqual(RecallPolicy.for_request("问题", "max_effort").timeout_seconds, 1800)
        self.assertEqual(RecallPolicy.for_request("问题", "deep", timeout_seconds=10).timeout_seconds, 10)
        self.assertEqual(RecallPolicy.for_request("问题", "deep", timeout_seconds=9999).timeout_seconds, 360)
        self.assertEqual(RecallPolicy.for_request("问题", "deep").timeout_seconds, 360)

    def test_only_explicit_maximum_imperative_routes_automatically(self):
        self.assertEqual(RecallPolicy.for_request("请尽最大努力回想那次事件").mode, "max_effort")
        self.assertTrue(requests_maximum_recall("Please try your best to recall the event"))
        self.assertFalse(requests_maximum_recall("不要尽最大努力回想"))
        self.assertFalse(requests_maximum_recall("‘尽最大努力回想’是什么意思？"))

    def test_invalid_timeout_and_mode_rejected(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RecallPolicy.for_request("q", timeout_seconds=value)
        with self.assertRaises(ValueError):
            RecallPolicy.for_request("q", mode="unlimited")

    def test_cli_exposes_resume_feedback_and_modes(self):
        parser = build_parser()
        args = parser.parse_args(["query", "q", "--mode", "max_effort", "--no-learn", "--resume", "a" * 32])
        self.assertEqual(args.mode, "max_effort")
        self.assertTrue(args.no_learn)
        feedback = parser.parse_args(["recall-feedback", "a" * 32, "negative", "--feedback-id", "feedback-1"])
        self.assertEqual(feedback.verdict, "negative")


if __name__ == "__main__":
    unittest.main()
