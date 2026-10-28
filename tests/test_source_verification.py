#!/usr/bin/env python3

import unittest
from pathlib import Path



from parse_ocr.verify_source_conflicts import (  # noqa: E402
    enforce_structured_checklist, independent_observer_prompt, parse_canonical_resolution,
    parse_resolution, resolve_canonical_consensus, resolve_from_blind_observers,
)


class SourceVerificationTests(unittest.TestCase):
    def setUp(self):
        self.conflict = {
            "conflict_set_id": "formula:abc", "candidate_ids": ["example:p1.x1", "example:p1.x2"]
        }

    def test_high_confidence_valid_resolution_is_applicable(self):
        raw = r"""<resolution>
<conflict_set_id>formula:abc</conflict_set_id>
<selected_id>example:p1.x2</selected_id>
<verified_statement>The value is $444^2$.</verified_statement>
<evidence>The crop prints 444.</evidence>
<confidence>high</confidence>
</resolution>"""
        result = parse_resolution(raw, self.conflict)
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["selected_id"], "example:p1.x2")

    def test_low_confidence_never_auto_applies(self):
        raw = """<resolution><conflict_set_id>formula:abc</conflict_set_id>
<selected_id>example:p1.x1</selected_id><verified_statement>unclear</verified_statement>
<evidence>cropped boundary</evidence><confidence>low</confidence></resolution>"""
        self.assertEqual(parse_resolution(raw, self.conflict)["status"], "unresolved")

    def test_high_confidence_canonical_context_decision_is_applicable(self):
        conflict = {"collision_set_id": "span:abc", "candidate_ids": ["notation:host", "notation:atom"]}
        raw = """<resolution><conflict_set_id>span:abc</conflict_set_id>
<action>context_only</action><selected_id>notation:atom</selected_id>
<context_only_ids>notation:host</context_only_ids><verified_statement></verified_statement>
<evidence>The host contains the complete atomic sentence plus unrelated setup.</evidence>
<confidence>high</confidence></resolution>"""
        result = parse_canonical_resolution(raw, conflict)
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["context_only_ids"], ["notation:host"])

    def test_medium_canonical_decision_never_auto_applies(self):
        conflict = {"collision_set_id": "span:abc", "candidate_ids": ["fact:a", "fact:b"]}
        raw = """<resolution><conflict_set_id>span:abc</conflict_set_id>
<action>retain_distinct</action><selected_id>fact:a</selected_id>
<context_only_ids></context_only_ids><verified_statement></verified_statement>
<evidence>Boundary is partly visible.</evidence><confidence>medium</confidence></resolution>"""
        self.assertEqual(parse_canonical_resolution(raw, conflict)["status"], "unresolved")

    def test_high_confidence_prose_cannot_bypass_missing_cell_checks(self):
        conflict = {
            "conflict_set_id": "transcription:table", "candidate_ids": ["example:table"],
            "verification_checklist": [
                {"id": "table:row=23:value", "candidate": "289993900455734779",
                 "requires_double_read": True},
                {"id": "table:row=25:value", "candidate": "4552540056626406637",
                 "requires_double_read": True},
            ],
        }
        raw = r"""<resolution><conflict_set_id>transcription:table</conflict_set_id>
<selected_id>example:table</selected_id>
<verified_statement>| $n$ | $A_n$ |
|---|---|
| $23$ | $289993909455734779$ |
| $25$ | $455254005662640637$ |</verified_statement>
<checks_json>[{"id":"table:row=23:value","observed_first":"289993909455734779","observed_second":"289993909455734779","legible":true}]</checks_json>
<evidence>I checked the table.</evidence><confidence>high</confidence></resolution>"""
        result = enforce_structured_checklist(parse_resolution(raw, conflict), conflict)
        self.assertEqual(result["status"], "unresolved")
        self.assertFalse(result["structured_verification"]["complete"])

    def test_novel_table_values_are_observed_but_never_auto_applied(self):
        conflict = {
            "conflict_set_id": "transcription:table", "candidate_ids": ["example:table"],
            "candidates": [{
                "id": "example:table",
                "statement": "| $n$ | $A_n$ |\n|---|---|\n| $23$ | $289993900455734779$ |\n| $25$ | $4552540056626406637$ |",
            }],
            "verification_checklist": [
                {"id": "table:row=23:value", "candidate": "289993900455734779",
                 "requires_double_read": True},
                {"id": "table:row=25:value", "candidate": "4552540056626406637",
                 "requires_double_read": True},
            ],
        }
        raw = r"""<resolution><conflict_set_id>transcription:table</conflict_set_id>
<selected_id>example:table</selected_id>
<verified_statement>| $n$ | $A_n$ |
|---|---|
| $23$ | $289993909455734779$ |
| $25$ | $455254005662640637$ |</verified_statement>
<checks_json>[
{"id":"table:row=23:value","observed_first":"289993909455734779","observed_second":"289993909455734779","legible":true},
{"id":"table:row=25:value","observed_first":"455254005662640637","observed_second":"455254005662640637","legible":true}
]</checks_json><evidence>cell-by-cell</evidence><confidence>high</confidence></resolution>"""
        result = enforce_structured_checklist(parse_resolution(raw, conflict), conflict)
        self.assertEqual(result["status"], "resolved")
        self.assertTrue(result["structured_verification"]["complete"])
        self.assertEqual(len(result["structured_verification"]["novel_observations"]), 2)
        consensus = resolve_from_blind_observers(conflict, [result, result, result])
        self.assertEqual(consensus["status"], "unresolved")
        self.assertEqual(consensus["reason"], "novel_observation_requires_human_verification")

    def test_blind_consensus_selects_existing_candidate_without_rewrite(self):
        conflict = {
            "conflict_set_id": "formula:abc", "candidate_ids": ["a", "b"],
            "candidates": [
                {"id": "a", "statement": "Printed $441^2$."},
                {"id": "b", "statement": "Printed $444^2$."},
            ],
            "verification_checklist": [{
                "id": "critical:0001", "role": "critical_token",
                "candidate_values": {"a": "441^2", "b": "444^2"},
            }],
        }
        observer = {
            "status": "resolved", "confidence": "high",
            "structured_verification": {"complete": True},
            "checks": [{"id": "critical:0001", "observed_first": "444^2"}],
        }
        dissenting = {
            "status": "resolved", "confidence": "high",
            "structured_verification": {"complete": True},
            "checks": [{"id": "critical:0001", "observed_first": "441^2"}],
        }
        result = resolve_from_blind_observers(conflict, [observer, dissenting, observer])
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["selected_id"], "b")
        self.assertEqual(result["verified_statement"], "Printed $444^2$.")
        self.assertFalse(result["audit_safety"]["model_generated_text_applied"])

    def test_three_distinct_blind_reads_leave_conflict_unresolved(self):
        conflict = {
            "conflict_set_id": "formula:abc", "candidate_ids": ["a", "b"],
            "candidates": [
                {"id": "a", "statement": "Printed $441$."},
                {"id": "b", "statement": "Printed $444$."},
            ],
            "verification_checklist": [{
                "id": "critical:0001", "role": "critical_token",
                "candidate_values": {"a": "441", "b": "444"},
            }],
        }
        observers = [{
            "status": "resolved", "confidence": "high",
            "structured_verification": {"complete": True},
            "checks": [{"id": "critical:0001", "observed_first": value}],
        } for value in ("441", "444", "447")]
        result = resolve_from_blind_observers(conflict, observers)
        self.assertEqual(result["status"], "unresolved")
        self.assertEqual(result["reason"], "blind_observer_consensus_incomplete")

    def test_mathematically_natural_p48_repair_is_not_an_existing_candidate(self):
        conflict = {
            "conflict_set_id": "formula:p48", "candidate_ids": ["printed"],
            "candidates": [{
                "id": "printed", "statement": r"The source prints $p_n^2>p_{n-1}p_{n+1}$ for all $i$.",
            }],
            "verification_checklist": [{
                "id": "critical:0001", "role": "critical_token",
                "candidate": r"p_n^2>p_{n-1}p_{n+1}",
            }],
        }
        hallucinated = {
            "status": "resolved", "confidence": "high",
            "structured_verification": {"complete": True},
            "checks": [{
                "id": "critical:0001", "observed_first": r"p_{n-i}p_{n+i}",
            }],
        }
        result = resolve_from_blind_observers(
            conflict, [hallucinated, hallucinated, hallucinated]
        )
        self.assertEqual(result["status"], "unresolved")
        self.assertEqual(result["reason"], "novel_observation_requires_human_verification")

    def test_canonical_consensus_ignores_generated_merge_text(self):
        conflict = {
            "collision_set_id": "span:abc", "candidate_ids": ["a", "b"],
            "candidates": [
                {"id": "a", "statement": "Literal source A."},
                {"id": "b", "statement": "Literal source B."},
            ],
        }
        agreeing = {
            "status": "resolved", "confidence": "high", "action": "merge",
            "selected_id": "b", "context_only_ids": [],
            "verified_statement": "Invented improved statement.",
        }
        result = resolve_canonical_consensus(conflict, [agreeing, agreeing, agreeing])
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["verified_statement"], "Literal source B.")

    def test_blind_prompt_contains_no_candidate_statement_or_values(self):
        conflict = {
            "conflict_set_id": "formula:abc", "candidate_ids": ["a", "b"],
            "candidates": [
                {"id": "a", "statement": "Secret $441^2$."},
                {"id": "b", "statement": "Secret $444^2$."},
            ],
            "verification_checklist": [{
                "id": "critical:0001", "role": "critical_token",
                "candidate_values": {"a": "441^2", "b": "444^2"},
            }],
        }
        value = independent_observer_prompt(conflict)
        self.assertNotIn("441", value)
        self.assertNotIn("444", value)
        self.assertNotIn("Secret", value)

if __name__ == "__main__":
    unittest.main()
