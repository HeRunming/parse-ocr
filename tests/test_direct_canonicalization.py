#!/usr/bin/env python3

import unittest
from pathlib import Path



from parse_ocr.direct_pipeline import (  # noqa: E402
    apply_source_span_resolutions,
    detect_canonical_text_overlaps,
    escape_literal_currency,
    structured_verification_checklist,
)


class DirectCanonicalizationTests(unittest.TestCase):
    def test_residual_contained_source_text_is_reported_not_auto_deleted(self):
        common = {
            "section_id": "A1",
            "source_spans": [{
                "pdf_page": 5, "block_start": 2, "block_end": 2,
                "token_start": 0, "token_end": 80,
            }],
        }
        host = {**common, "id": "remark:r1", "statement": (
            "The game loops. "
            "For example from two the next player goes to three and then returns to two."
        )}
        child = {**common, "id": "example:x1", "statement": (
            "For example from two the next player goes to three and then returns to two."
        )}
        overlaps = detect_canonical_text_overlaps([host, child])
        self.assertEqual(len(overlaps), 1)
        self.assertEqual(overlaps[0]["status"], "unresolved")
        self.assertEqual(len([host, child]), 2)

    def test_declared_list_count_mismatch_is_high_risk(self):
        checklist, reasons = structured_verification_checklist({
            "statement": "The first 10 values are 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37.",
            "uncertain_spans": [],
        })
        self.assertTrue(checklist)
        self.assertIn("dense_numeric_sequence", reasons)
        self.assertIn("declared_item_count_mismatch:10!=11", reasons)

    def test_currency_is_escaped_but_math_dollars_are_unchanged(self):
        self.assertEqual(
            escape_literal_currency(r"Offers $500.00 for a proof; $2.33$ is a bound on $x$."),
            r"Offers \$500.00 for a proof; $2.33$ is a bound on $x$.",
        )

    def test_context_resolution_is_removed_from_canonical_and_remapped(self):
        rows = [
            {"id": "exercise:e1", "merged_source_ids": []},
            {"id": "notation:n1", "merged_source_ids": []},
        ]
        collisions = [{
            "collision_set_id": "span:abc", "status": "unresolved",
            "candidate_ids": ["exercise:e1", "notation:n1"],
        }]
        resolutions = [{
            "conflict_set_id": "span:abc", "status": "resolved", "confidence": "high",
            "action": "context_only", "selected_id": "exercise:e1",
            "context_only_ids": ["notation:n1"], "evidence": "visible source boundary",
            "audit_safety": {
                "evidence_protocol": "v6_existing_candidate_only",
                "model_generated_text_applied": False,
            },
            "canonical_consensus": {"complete": True},
        }]
        kept, context, superseded, remap = apply_source_span_resolutions(
            rows, collisions, resolutions
        )
        self.assertEqual([row["id"] for row in kept], ["exercise:e1"])
        self.assertEqual(context[0]["canonical_role"], "context_only")
        self.assertEqual(superseded, [])
        self.assertEqual(remap, {"notation:n1": "exercise:e1"})
        self.assertEqual(collisions[0]["status"], "resolved")

    def test_legacy_single_model_canonical_resolution_cannot_apply(self):
        rows = [
            {"id": "exercise:e1", "merged_source_ids": []},
            {"id": "notation:n1", "merged_source_ids": []},
        ]
        collisions = [{
            "collision_set_id": "span:legacy", "status": "unresolved",
            "candidate_ids": ["exercise:e1", "notation:n1"],
        }]
        legacy = [{
            "conflict_set_id": "span:legacy", "status": "resolved", "confidence": "high",
            "action": "context_only", "selected_id": "exercise:e1",
            "context_only_ids": ["notation:n1"],
        }]
        kept, context, superseded, remap = apply_source_span_resolutions(
            rows, collisions, legacy
        )
        self.assertEqual([row["id"] for row in kept], ["exercise:e1", "notation:n1"])
        self.assertEqual(context, [])
        self.assertEqual(superseded, [])
        self.assertEqual(remap, {})

    def test_v6_machine_transcription_cannot_rewrite_existing_candidate(self):
        from parse_ocr.direct_pipeline import apply_transcription_resolutions

        rows = [{"id": "fact:f1", "statement": "Printed awkward $n$.", "uncertain_spans": ["n"]}]
        resolutions = [{
            "selected_id": "fact:f1", "status": "resolved", "confidence": "high",
            "verified_statement": "Mathematically improved $i$.",
            "audit_safety": {
                "evidence_protocol": "v6_existing_candidate_only",
                "model_generated_text_applied": False,
            },
        }]
        report = apply_transcription_resolutions(rows, resolutions)
        self.assertEqual(report["applied"], 0)
        self.assertEqual(rows[0]["statement"], "Printed awkward $n$.")

    def test_transcription_verifier_cannot_expand_into_neighboring_objects(self):
        from parse_ocr.direct_pipeline import apply_transcription_resolutions

        rows = [{"id": "fact:f1", "statement": "A short fact.", "uncertain_spans": ["x"]}]
        resolutions = [{
            "selected_id": "fact:f1", "status": "resolved", "confidence": "high",
            "verified_statement": "A short fact. " + "Neighboring material. " * 20,
            "human_pdf_visual_verification": True,
        }]
        report = apply_transcription_resolutions(rows, resolutions)
        self.assertEqual(report["applied"], 0)
        self.assertEqual(rows[0]["statement"], "A short fact.")
        self.assertEqual(len(report["rejected_boundary_expansion"]), 1)

    def test_transcription_verifier_cannot_absorb_neighbor_with_small_length_growth(self):
        from parse_ocr.direct_pipeline import apply_transcription_resolutions

        rows = [
            {"id": "definition:d1", "section_id": "A1", "source_pdf_page": 5,
             "statement": "Define the sequence a_n by a recurrence.", "uncertain_spans": ["a_n"]},
            {"id": "conjecture:c1", "section_id": "A1", "source_pdf_page": 5,
             "statement": "It is conjectured that every term of the sequence is prime.",
             "uncertain_spans": []},
        ]
        resolutions = [{
            "selected_id": "definition:d1", "status": "resolved", "confidence": "high",
            "verified_statement": (
                "Define the sequence a_n by a recurrence. "
                "It is conjectured that every term of the sequence is prime."
            ),
            "human_pdf_visual_verification": True,
        }]
        report = apply_transcription_resolutions(rows, resolutions)
        self.assertEqual(report["applied"], 0)
        self.assertEqual(
            report["rejected_boundary_expansion"][0]["absorbed_neighbor_id"],
            "conjecture:c1",
        )


if __name__ == "__main__":
    unittest.main()
