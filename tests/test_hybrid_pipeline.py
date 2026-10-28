#!/usr/bin/env python3

import unittest
from pathlib import Path



from parse_ocr.hybrid_pipeline import (  # noqa: E402
    apply_atomicity_refinements,
    apply_reviewed_audit_overrides,
    attach_indexed_figure_assets,
    atomic_sentence_units,
    complete_source_context_groups,
    factor_hybrid_context_only,
    figure_asset_index,
    parse_atomicity_refinement,
    repair_secondary_lexical_typos,
    repair_source_confirmed_linebreak_hyphens,
    repair_source_confirmed_long_integer_spacing,
    sanitize_transport_control_artifacts,
    strip_tex_image_commands,
    stabilize_physical_source_order,
)
from parse_ocr.source_grounding import populate_context  # noqa: E402


class HybridPipelineTests(unittest.TestCase):
    def test_lexical_repair_requires_native_source_evidence(self):
        repaired, events = repair_secondary_lexical_typos(
            "Fouvry proved the result.",
            "Fourry proved the result.",
            "The printed source says Fouvry proved the result.",
        )
        self.assertEqual(repaired, "Fouvry proved the result.")
        self.assertEqual(events, [])

    def test_lexical_repair_fixes_source_confirmed_ocr_typo(self):
        repaired, events = repair_secondary_lexical_typos(
            "This is satisfied by tbe Queens and tbe Bishops.",
            "This is satisfied by the Queens and the Bishops.",
            "This is satisfied by the Queens and the Bishops.",
        )
        self.assertEqual(repaired, "This is satisfied by the Queens and the Bishops.")
        self.assertEqual(events[0]["from"], "tbe")

    def test_accent_only_name_repair_is_allowed(self):
        repaired, _ = repair_secondary_lexical_typos(
            "Erdös proved it.", "Erdős proved it.", "Erdos proved it."
        )
        self.assertEqual(repaired, "Erdős proved it.")

    def test_anaphoric_atomic_child_gets_exact_preceding_context(self):
        rows = [
            {
                "id": "definition:A.001", "section_id": "A", "statement": "Define M as the maximum.",
                "source_order_key": [1, 1, 0, 0],
            },
            {
                "id": "notation:A.002", "section_id": "A",
                "statement": "Denote by h(x) the corresponding maximum.",
                "source_order_key": [1, 2, 0, 0],
            },
        ]
        populate_context(rows)
        self.assertEqual(rows[1]["context_before_id"], "definition:A.001")
        self.assertEqual(rows[1]["context_before"], rows[0]["statement"])

    def test_native_backed_linebreak_hyphen_is_joined(self):
        repaired, events = repair_source_confirmed_linebreak_hyphens(
            "a pseudoper-fect number", "a pseudoperfect number", "a pseudoperfect number"
        )
        self.assertEqual(repaired, "a pseudoperfect number")
        self.assertEqual(len(events), 1)

    def test_native_backed_long_integer_is_joined(self):
        repaired, _ = repair_source_confirmed_long_integer_spacing(
            "y=11817143185 2779451900", "y = 118171431852779451900"
        )
        self.assertEqual(repaired, "y=118171431852779451900")

    def test_atomic_split_is_exact_and_preserves_parent_context(self):
        row = {
            "id": "conjecture:p1.c1", "kind": "conjecture", "label": "p1.c1",
            "section_id": "A1", "statement": "Is the set A always finite? Is the set B always finite?",
            "entry_id": "A1.001", "canonical_source_id": "source:1",
            "source_order_key": [1, 1, 0, 0], "confidence": {},
        }
        units = atomic_sentence_units(row["statement"])
        decision = parse_atomicity_refinement(
            "<decision>split</decision>"
            "<part><kind>conjecture</kind><start>U001</start><end>U001</end></part>"
            "<part><kind>conjecture</kind><start>U002</start><end>U002</end></part>",
            row, units,
        )
        children, parents, _ = apply_atomicity_refinements([row], [decision])
        self.assertEqual(
            [child["statement"] for child in children],
            ["Is the set A always finite? ", "Is the set B always finite?"],
        )
        self.assertTrue(all(child["source_context"] == row["statement"] for child in children))
        self.assertEqual(parents[0]["canonical_role"], "context_only")

    def test_atomic_split_rejects_unbalanced_math_delimiter(self):
        row = {
            "id": "conjecture:p1.c1", "kind": "conjecture", "label": "p1.c1",
            "section_id": "A1", "statement": "Is $A finite? Is B$ finite?",
        }
        decision = parse_atomicity_refinement(
            "<decision>split</decision>"
            "<part><kind>conjecture</kind><start>U001</start><end>U001</end></part>"
            "<part><kind>conjecture</kind><start>U002</start><end>U002</end></part>",
            row, atomic_sentence_units(row["statement"]),
        )
        self.assertFalse(decision["accepted"])

    def test_image_transport_markup_is_removed(self):
        self.assertEqual(
            strip_tex_image_commands(r"\begin{center}\includegraphics{x.png}\end{center}"), ""
        )

    def test_reviewed_override_is_hash_guarded(self):
        import hashlib

        row = {"id": "fact:p1.f1", "statement": "old", "section_id": "A1"}
        value = {
            "updates": [{
                "id": row["id"],
                "expected_statement_sha256": hashlib.sha256(b"old").hexdigest(),
                "statement": "new",
                "audit_id": "test",
            }]
        }
        rows, _, _, audit = apply_reviewed_audit_overrides([row], [], [], value)
        self.assertEqual(rows[0]["statement"], "new")
        self.assertEqual(audit[0]["status"], "applied")

        stale = {"updates": [{**value["updates"][0], "statement": "bad"}]}
        rows, _, _, audit = apply_reviewed_audit_overrides([row], [], [], stale)
        self.assertEqual(rows[0]["statement"], "new")
        self.assertEqual(audit[0]["status"], "rejected_stale_or_missing_target")

    def test_reviewed_override_can_repair_context_parent_and_multiple_pages(self):
        import hashlib

        parent = {
            "id": "conjecture:p1.c1", "statement": "wrong", "section_id": "A1",
            "source_pages": [1], "source_spans": [], "statement_sources": [],
        }
        value = {
            "updates": [{
                "id": parent["id"],
                "expected_statement_sha256": hashlib.sha256(b"wrong").hexdigest(),
                "statement": "correct across pages",
                "pdf_pages": [1, 2],
                "audit_id": "test-context",
            }],
            "rejected_findings": [{
                "id": "definition:p2.d1", "audit_id": "false-alarm",
                "evidence": "The PDF preserves the original reading.",
            }],
        }
        _, contexts, _, audit = apply_reviewed_audit_overrides([], [parent], [], value)
        self.assertEqual(contexts[0]["statement"], "correct across pages")
        self.assertEqual(contexts[0]["source_pages"], [1, 2])
        self.assertEqual({span["pdf_page"] for span in contexts[0]["source_spans"]}, {1, 2})
        self.assertEqual(audit[-1]["status"], "rejected_no_writeback")

    def test_reviewed_override_exact_field_replacement(self):
        import hashlib

        row = {
            "id": "exercise:p1.e1", "statement": "Question.",
            "supporting_text": "Use 2\\zeta\\eta(\\xi,\\eta,\\zeta).",
        }
        value = {"updates": [{
            "id": row["id"],
            "expected_statement_sha256": hashlib.sha256(b"Question.").hexdigest(),
            "replace_fields": {"supporting_text": [{
                "old": "2\\zeta\\eta", "new": "2\\xi\\eta\\zeta", "count": 1,
            }]},
        }]}
        rows, _, _, audit = apply_reviewed_audit_overrides([row], [], [], value)
        self.assertIn("2\\xi\\eta\\zeta", rows[0]["supporting_text"])
        self.assertEqual(audit[0]["status"], "applied")

    def test_transport_control_artifacts_are_removed(self):
        rows = [{
            "id": "definition:p1.d1", "entry_id": "A1.001",
            "statement": "<![CDATA[Define $f(n)$.]]>",
            "supporting_text": "</supporting>", "supporting_text_role": "answer",
        }]
        events = sanitize_transport_control_artifacts(rows)
        self.assertEqual(rows[0]["statement"], "Define $f(n)$.")
        self.assertEqual(rows[0]["supporting_text"], "")
        self.assertEqual(rows[0]["supporting_text_role"], "")
        self.assertEqual(len(events), 2)

    def test_reviewed_source_verified_addition_is_explicit_and_audited(self):
        host = {
            "id": "definition:p1.d1", "entry_id": "A1.001", "kind": "definition",
            "statement": "Define the displayed sequence.", "canonical_role": "canonical",
        }
        value = {
            "source_verified_additions": [{
                "id": "conjecture:reviewed.A1.c001a",
                "entry_id": "A1.001a",
                "kind": "conjecture",
                "label": "reviewed.A1.c001a",
                "section_id": "A1",
                "statement": "Is the displayed limit zero?",
                "supporting_text": "It is zero for almost all inputs.",
                "supporting_text_role": "answer",
                "pdf_page": 2,
                "source_order_key": [1, 2_000_000, 1, 1],
                "audit_id": "blind-test",
                "source_context_target_id": host["id"],
            }]
        }
        rows, _, _, audit = apply_reviewed_audit_overrides([host], [], [], value)
        self.assertEqual(rows[1]["entry_id"], "A1.001a")
        self.assertTrue(rows[1]["source_verified_addition"])
        self.assertEqual(rows[1]["supporting_text_role"], "answer")
        self.assertEqual(rows[1]["source_context"], host["statement"])
        self.assertEqual(rows[1]["source_context_host_id"], host["id"])
        self.assertEqual(audit[0]["status"], "added_source_verified")

    def test_reviewed_context_only_promotion_is_hash_guarded_and_removes_event(self):
        import hashlib

        context = {
            "id": "example:p2.x1", "entry_id": "A1.002", "kind": "example",
            "statement": "$$1,2,3$$", "canonical_role": "context_only",
            "context_only_reason": "bare_display_supports_previous_setup",
        }
        events = [{"context_only_id": context["id"], "reason": "bare display"}]
        value = {"promote_context_only": [{
            "id": context["id"],
            "expected_statement_sha256": hashlib.sha256(context["statement"].encode()).hexdigest(),
            "set_fields": {"reviewed_source_order_rank": 2.5},
            "audit_id": "blind-test-promotion",
            "evidence": "The rendered page shows a standalone example table.",
        }]}
        rows, contexts, events, audit = apply_reviewed_audit_overrides(
            [], [context], events, value,
        )
        self.assertEqual([row["id"] for row in rows], [context["id"]])
        self.assertEqual(contexts, [])
        self.assertEqual(events, [])
        self.assertEqual(rows[0]["canonical_role"], "canonical")
        self.assertNotIn("context_only_reason", rows[0])
        self.assertEqual(rows[0]["reviewed_source_order_rank"], 2.5)
        self.assertEqual(audit[0]["status"], "promoted_from_context_only")

    def test_reviewed_promotion_runs_after_reviewed_context_only_action(self):
        import hashlib

        row = {
            "id": "claim:p3.c1", "entry_id": "A1.003", "kind": "claim",
            "statement": "This holds.", "canonical_role": "canonical",
            "section_id": "A1",
        }
        digest = hashlib.sha256(row["statement"].encode()).hexdigest()
        value = {
            "context_only": [{
                "id": row["id"], "expected_statement_sha256": digest,
                "canonical_target_id": "", "audit_id": "factor-first",
            }],
            "promote_context_only": [{
                "id": row["id"], "expected_statement_sha256": digest,
                "statement": "The displayed construction has the claimed property.",
                "audit_id": "promote-second",
            }],
        }
        rows, contexts, events, audit = apply_reviewed_audit_overrides(
            [row], [], [], value,
        )
        self.assertEqual(contexts, [])
        self.assertEqual(events, [])
        self.assertEqual(rows[0]["statement"], "The displayed construction has the claimed property.")
        self.assertEqual(rows[0]["statement_before_reviewed_override"], "This holds.")
        self.assertEqual([item["status"] for item in audit], [
            "moved_to_context_only", "promoted_from_context_only",
        ])

    def test_reviewed_page_replacement_drops_disproved_old_localization(self):
        import hashlib

        row = {
            "id": "theorem:wrong-page", "entry_id": "A1.001", "kind": "theorem",
            "statement": "Visible theorem", "section_id": "A1",
            "source_pdf_page": 82, "source_pages": [82],
            "source_spans": [{"source_span_id": "p82.old", "pdf_page": 82}],
            "statement_sources": [{"pdf_page": 82, "method": "old"}],
        }
        value = {"updates": [{
            "id": row["id"],
            "expected_statement_sha256": hashlib.sha256(row["statement"].encode()).hexdigest(),
            "pdf_page": 81, "replace_source_pages": True,
            "audit_id": "blind:test-page",
        }]}
        rows, _, _, _ = apply_reviewed_audit_overrides([row], [], [], value)
        self.assertEqual(rows[0]["source_pdf_page"], 81)
        self.assertEqual(rows[0]["source_pages"], [81])
        self.assertTrue(all(span["pdf_page"] == 81 for span in rows[0]["source_spans"]))
        self.assertEqual(rows[0]["source_pages_before_reviewed_override"], [82])

    def test_context_only_entry_id_cannot_collide_with_canonical(self):
        import hashlib

        canonical = {
            "id": "theorem:canonical", "entry_id": "A1.001", "kind": "theorem",
            "statement": "A theorem", "section_id": "A1",
        }
        losing = {
            "id": "theorem:losing", "entry_id": "A1.001", "kind": "theorem",
            "statement": "A duplicate theorem", "section_id": "A1",
        }
        value = {"context_only": [{
            "id": losing["id"],
            "expected_statement_sha256": hashlib.sha256(losing["statement"].encode()).hexdigest(),
            "canonical_target_id": canonical["id"],
            "audit_id": "blind:test-collision",
        }]}
        rows, contexts, _, _ = apply_reviewed_audit_overrides(
            [canonical, losing], [], [], value,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(contexts[0]["candidate_entry_id"], canonical["entry_id"])
        self.assertTrue(contexts[0]["entry_id"].startswith("context_only:"))
        self.assertNotEqual(contexts[0]["entry_id"], canonical["entry_id"])

    def test_article_level_figure_assets_attach_to_direct_consumer(self):
        articles = [{
            "body_tex": "\\includegraphics{images/10_left.jpg}\n\n"
            "\\includegraphics{images/10_right.jpg}\n\n"
            "Figure 14(a). Left configuration. Figure 14(b). Right configuration."
        }]
        index = figure_asset_index(articles)
        rows = [{"statement": "Use the configuration in Fig. 14(b).", "assets": []}]
        attach_indexed_figure_assets(rows, index)
        self.assertEqual(rows[0]["assets"][0]["image"], "assets/10_right.jpg")
        self.assertEqual(rows[0]["assets"][0]["pdf_page"], 11)

    def test_reviewed_figure_asset_wins_over_secondary_duplicate(self):
        rows = [{
            "statement": "Use Figure 4.",
            "assets": [{
                "image": "assets/reviewed-figure4.png",
                "caption": "Figure 4. Reviewed crop.",
            }],
        }]
        attach_indexed_figure_assets(rows, {
            "4": {"image": "assets/secondary-figure4.jpg", "caption": "Figure 4. Export."}
        })
        self.assertEqual(len(rows[0]["assets"]), 1)
        self.assertEqual(rows[0]["assets"][0]["image"], "assets/reviewed-figure4.png")

    def test_physical_page_order_overrides_noisy_article_order(self):
        rows = [
            {"id": "a", "section_id": "A1", "source_pdf_page": 12, "source_order_key": [1, 1]},
            {"id": "b", "section_id": "A1", "source_pdf_page": 11, "source_order_key": [1, 2]},
        ]
        stabilize_physical_source_order(rows)
        self.assertLess(tuple(rows[1]["source_order_key"]), tuple(rows[0]["source_order_key"]))

    def test_reviewed_same_page_rank_is_stable_across_rebuilds(self):
        rows = [
            {
                "id": "before", "section_id": "A1", "source_pdf_page": 12,
                "source_order_key": [1, 12_000_000, 2, 0],
                "reviewed_source_order_rank": 20,
            },
            {
                "id": "inserted", "section_id": "A1", "source_pdf_page": 12,
                "source_order_key": [1, 12_000_000, 99, 0],
                "reviewed_source_order_rank": 25,
            },
            {
                "id": "after", "section_id": "A1", "source_pdf_page": 12,
                "source_order_key": [1, 12_000_000, 3, 0],
                "reviewed_source_order_rank": 30,
            },
        ]
        stabilize_physical_source_order(rows)
        first = [tuple(row["source_order_key"]) for row in rows]
        stabilize_physical_source_order(rows)
        second = [tuple(row["source_order_key"]) for row in rows]
        self.assertEqual(first, second)
        self.assertLess(first[0], first[1])
        self.assertLess(first[1], first[2])

    def test_substantive_assertion_after_display_remains_canonical(self):
        rows = [{
            "id": "example:test", "entry_id": "A1.1", "kind": "example",
            "section_id": "A1", "statement": "\\[1,2,3\\] is a 3-cycle.",
            "context_before_id": "definition:test", "source_order_key": [1, 2],
        }]
        canonical, context_only, _events = factor_hybrid_context_only(rows)
        self.assertEqual([row["id"] for row in canonical], ["example:test"])
        self.assertEqual(context_only, [])

    def test_missing_row_context_group_is_materialized_with_entry_membership(self):
        rows = [{
            "id": "fact:p1.f1", "entry_id": "A1.001", "section_id": "A1",
            "source_pdf_page": 1, "source_context_group_id": "src:p1.b2-2",
            "source_position": {"block_start": 2, "block_end": 2, "bbox": [1, 2, 3, 4]},
        }]
        groups = complete_source_context_groups(rows, [])
        self.assertEqual(groups[0]["group_id"], "src:p1.b2-2")
        self.assertEqual(groups[0]["entry_ids"], ["A1.001"])
        self.assertEqual(groups[0]["source_ids"], ["fact:p1.f1"])

    def test_existing_context_group_source_ids_get_entry_ids(self):
        rows = [
            {"id": "example:p1.x1", "entry_id": "A1.001", "section_id": "A1"},
            {"id": "remark:p1.r1", "entry_id": "A1.002", "section_id": "A1"},
        ]
        groups = complete_source_context_groups(rows, [{
            "group_id": "ctx:example_p1.x1",
            "source_ids": ["example:p1.x1", "remark:p1.r1"],
        }])
        self.assertEqual(groups[0]["entry_ids"], ["A1.001", "A1.002"])


if __name__ == "__main__":
    unittest.main()
