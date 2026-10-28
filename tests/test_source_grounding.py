#!/usr/bin/env python3

import unittest
from pathlib import Path



from parse_ocr.source_grounding import (  # noqa: E402
    TextBlock,
    align_text_to_page,
    anchor_sections,
    apply_formula_resolutions,
    assign_sections,
    attach_statement_completions,
    detect_formula_conflicts,
    factor_repeated_prefix_context,
    formula_tokens,
    merge_format_duplicates,
    merge_near_duplicates,
    merge_sentence_fragments,
    populate_source_context_groups,
    semantic_fingerprint,
    structured_math_items,
)


def row(item_id, kind, statement, block, token_start=0, token_end=20):
    return {
        "id": item_id,
        "kind": kind,
        "label": item_id.split(":", 1)[1],
        "statement": statement,
        "supporting_text": "",
        "supporting_text_role": "",
        "source_pdf_page": 10,
        "source_pages": [10],
        "source_position": {
            "page": 10, "block_start": block, "block_end": block,
            "bbox": [10, 100 + block * 20, 300, 115 + block * 20],
            "token_start": token_start, "token_end": token_end, "score": 0.9,
        },
        "source_order_key": [10, 100 + block * 20, block, token_start],
        "uncertain_spans": [],
    }


class SourceGroundingTests(unittest.TestCase):
    def test_semantic_fingerprint_ignores_presentation_and_casefolds(self):
        self.assertEqual(
            semantic_fingerprint("**Gauß** gives $\\mathbf{x}$"),
            semantic_fingerprint("Gauss gives $x$"),
        )
        self.assertIn("cilleruelo", semantic_fingerprint("$A(x)<c\\sqrt{x}$. Cilleruelo obtains $r>2$."))

    def test_alignment_prefers_smallest_equal_scoring_interval(self):
        blocks = [
            TextBlock(1, 1, (10, 50, 300, 80), "Unrelated preceding prose", 400, 600),
            TextBlock(1, 2, (10, 90, 300, 120), "For example 70 is not the sum of any subset of", 400, 600),
        ]
        result = align_text_to_page("For example, 70 is not the sum of any subset of", blocks)
        self.assertEqual((result["block_start"], result["block_end"]), (2, 2))

    def test_dedup_requires_same_source_span(self):
        first = row("example:p10.x1", "example", "An **example** is $x^2$.", 2)
        duplicate = row("example:p10.x2", "example", "An example is $x^2$.", 2)
        repeated_elsewhere = row("example:p10.x3", "example", "An example is $x^2$.", 8)
        values, remap, events = merge_format_duplicates([first, duplicate, repeated_elsewhere])
        self.assertEqual(len(values), 2)
        self.assertEqual(len(remap), 1)
        self.assertEqual(len(events), 1)
        self.assertIn("example:p10.x3", {value["id"] for value in values})

    def test_near_duplicate_typo_is_merged_with_auditable_roles(self):
        typo = row(
            "fact:p10.f1", "fact",
            r"Bateman, Erdős, Pomerance & Strauss show that $A$ has density 1", 2,
        )
        source = row(
            "fact:p10.f2", "fact",
            r"Bateman, Erdős, Pomerance & Straus show that $A$ has density 1.", 2,
        )
        typo["section_id"] = source["section_id"] = "A1"
        values, remap, events = merge_near_duplicates([typo, source])
        self.assertEqual(len(values), 1)
        self.assertEqual(len(remap), 1)
        self.assertEqual(len(events), 1)
        roles = {candidate["canonical_role"] for candidate in values[0]["candidate_variants"]}
        self.assertEqual(roles, {"canonical", "superseded"})

    def test_near_duplicate_does_not_merge_math_critical_disagreement(self):
        indexed = row(
            "conjecture:p10.c1", "conjecture",
            r"Is it true that $x_{i_k}$ divides $a_k$ for every $k$?", 2,
        )
        unindexed = row(
            "example:p10.x1", "example",
            r"Is it true that $x_i$ divides $a_k$ for every $k$?", 2,
        )
        indexed["section_id"] = unindexed["section_id"] = "A1"
        values, remap, events = merge_near_duplicates([indexed, unindexed])
        self.assertEqual(len(values), 2)
        self.assertEqual(remap, {})
        self.assertEqual(events, [])

    def test_fragment_merge_is_same_kind_and_source_sentence_only(self):
        left = row("fact:p10.f1", "fact", "Four sequences end at 612217", 2, 0, 10)
        right = row("fact:p10.f2", "fact", "and a sequence of 11 ends at 766373.", 2, 10, 20)
        alien = row("example:p10.x1", "example", "the unrelated example.", 2, 20, 25)
        values, remap, _ = merge_sentence_fragments([left, right, alien])
        self.assertEqual(len(values), 2)
        self.assertEqual(remap, {"fact:p10.f2": "fact:p10.f1"})
        self.assertIn("and a sequence", values[0]["statement"])

    def test_display_completing_sentence_is_not_answer(self):
        value = row("example:p10.x1", "example", "A specimen is", 2)
        value.update({"supporting_text": "$$2^3=8$$", "supporting_text_role": "answer",
                      "supporting_text_origin": "book_source"})
        attach_statement_completions([value])
        self.assertIn("$$2^3=8$$", value["statement"])
        self.assertEqual(value["supporting_text"], "")

    def test_section_assignment_uses_within_page_heading_order(self):
        before = row("fact:p10.f1", "fact", "Before heading", 2)
        after = row("fact:p10.f2", "fact", "After heading", 7)
        anchors = [
            {"label": "A1", "source_order_key": [9, 100, 2, 0], "source_position": {"score": 1}},
            {"label": "A2", "source_order_key": [10, 200, 5, 0], "source_position": {"score": 1}},
        ]
        assign_sections([before, after], anchors)
        self.assertEqual(before["section_id"], "A1")
        self.assertEqual(after["section_id"], "A2")

    def test_heading_above_running_header_cutoff_is_not_discarded(self):
        blocks = {
            10: [
                TextBlock(10, 0, (10, 8, 300, 19), "B. Divisibility", 400, 600),
                TextBlock(10, 1, (10, 21, 300, 55), "B38 Solutions of phi(m)=sigma(n).", 400, 600),
                TextBlock(10, 8, (10, 470, 300, 500), "An example with n=33817088", 400, 600),
            ]
        }
        anchors = anchor_sections([
            {"label": "B38", "title": "Solutions of phi(m)=sigma(n)", "start_page": 10},
        ], blocks)
        self.assertEqual(anchors[0]["source_position"]["block_start"], 1)
        self.assertEqual(anchors[0]["source_position"]["bbox"][1], 21)

    def test_contained_atomic_object_keeps_context_without_being_deleted(self):
        host = row("conjecture:p10.c1", "conjecture", "Define F. It is conjectured that F is finite.", 2)
        atom = row("definition:p10.d1", "definition", "Define F.", 2)
        host["section_id"] = atom["section_id"] = "A1"
        groups = populate_source_context_groups([host, atom])
        self.assertEqual(len(groups), 1)
        self.assertEqual(atom["source_context_host_id"], host["id"])
        self.assertIn("conjectured", atom["source_context"])

    def test_formula_tokens_ignore_matrix_presentation(self):
        self.assertEqual(
            formula_tokens(r"$$\begin{bmatrix}1&2\\3&4\end{bmatrix}$$"),
            formula_tokens(r"$$\left(\begin{array}{cc}1&2\\3&4\end{array}\right)$$"),
        )

    def test_same_span_critical_formula_difference_is_hard_conflict(self):
        wrong = row(
            "example:p10.x1", "example",
            r"The array is $$\begin{bmatrix}111^2&222^2\\333^2&441^2\end{bmatrix}.$$", 2,
        )
        correct = row(
            "example:p10.x2", "example",
            r"The array is $$\begin{bmatrix}111^2&222^2\\333^2&444^2\end{bmatrix}.$$", 2,
        )
        wrong["section_id"] = correct["section_id"] = "A1"
        conflicts = detect_formula_conflicts([wrong, correct])
        self.assertEqual(len(conflicts), 1)
        self.assertTrue(conflicts[0]["hard_conflict"])
        self.assertIn("441", str(conflicts[0]["critical_differences"]))
        self.assertEqual(len(conflicts[0]["verification_checklist"]), 1)
        self.assertEqual(
            set(conflicts[0]["verification_checklist"][0]["candidate_values"].values()),
            {"441", "444"},
        )

        resolution = [{
            "conflict_set_id": conflicts[0]["conflict_set_id"], "status": "resolved",
            "selected_id": correct["id"], "verified_statement": correct["statement"],
            "confidence": "high", "evidence": "crop visibly prints 444^2",
            "structured_verification": {"complete": True},
            "independent_observer": {"complete": True},
            "audit_safety": {
                "evidence_protocol": "v6_existing_candidate_only",
                "model_generated_text_applied": False,
            },
        }]
        values, remap, applied = apply_formula_resolutions(
            [wrong, correct], conflicts, resolution
        )
        self.assertEqual([value["id"] for value in values], [correct["id"]])
        self.assertEqual(remap, {wrong["id"]: correct["id"]})
        self.assertEqual(len(applied), 1)
        self.assertEqual(values[0]["canonical_status"], "source_verified")

    def test_high_canonical_crop_audit_closes_matching_formula_ledger(self):
        wrong = row(
            "example:p10.x1", "example",
            r"The array is $$\begin{bmatrix}111^2&222^2\\333^2&441^2\end{bmatrix}.$$", 2,
        )
        correct = row(
            "example:p10.x2", "example",
            r"The array is $$\begin{bmatrix}111^2&222^2\\333^2&444^2\end{bmatrix}.$$", 2,
        )
        wrong["section_id"] = correct["section_id"] = "A1"
        conflict = detect_formula_conflicts([wrong, correct])[0]
        canonical_audit = [{
            "conflict_set_id": conflict["conflict_set_id"],
            "status": "resolved", "confidence": "high", "action": "merge",
            "selected_id": correct["id"], "verified_statement": correct["statement"],
            "evidence": "The independent source crop visibly reads 444^2.",
            "canonical_source_verification": True,
            "audit_safety": {
                "evidence_protocol": "v6_existing_candidate_only",
                "model_generated_text_applied": False,
            },
            "canonical_consensus": {"complete": True},
        }]
        values, remap, applied = apply_formula_resolutions(
            [wrong, correct], [conflict], canonical_audit
        )
        self.assertEqual([value["id"] for value in values], [correct["id"]])
        self.assertEqual(remap, {wrong["id"]: correct["id"]})
        self.assertEqual(len(applied), 1)

    def test_formula_conflict_requires_overlapping_source_span(self):
        first = row("example:p10.x1", "example", r"The value is $441^2$.", 2)
        second = row("example:p10.x2", "example", r"The value is $444^2$.", 8)
        first["section_id"] = second["section_id"] = "A1"
        self.assertEqual(detect_formula_conflicts([first, second]), [])

    def test_adjacent_token_ranges_in_one_native_block_are_distinct(self):
        first = row("conjecture:p10.c1", "conjecture", r"Does $S$ contain odd numbers?", 2, 10, 20)
        second = row("conjecture:p10.c2", "conjecture", r"Does $M$ contain odd numbers?", 2, 20, 30)
        first["section_id"] = second["section_id"] = "A1"
        self.assertEqual(detect_formula_conflicts([first, second]), [])

    def test_b43_cross_candidate_table_conflict_is_cell_level(self):
        wrong = row(
            "example:p116.x2", "example",
            r"""$$\begin{array}{c|l@{\qquad}c|l}
n&A_n&n&A_n\\
9&79\cdot4139&23&85439\cdot289993900455734779\\
10&3301819&25&59\cdot555307\cdot4552540056626406637\\
11&13\cdot2816537&28&61\cdot221171\cdot2182035775749410439949\\
12&29\cdot15254711&24&12203\cdot24281
\end{array}$$""", 2,
        )
        correct = row(
            "notation:p116.d1", "notation",
            r"""| $n$ | $A_n$ |
|---|---|
| $9$ | $79 \cdot 4139$ |
| $10$ | $3301819$ |
| $11$ | $13 \cdot 2816537$ |
| $12$ | $29 \cdot 15254711$ |
| $23$ | $85439 \cdot 289993909455734779$ |
| $24$ | $12203 \cdot 24281$ |
| $25$ | $59 \cdot 555307 \cdot 455254005662640637$ |
| $28$ | $61 \cdot 221171 \cdot 21820357757749410439949$ |""", 2,
        )
        wrong["section_id"] = correct["section_id"] = "B43"
        conflicts = detect_formula_conflicts([wrong, correct])
        self.assertEqual(len(conflicts), 1)
        conflict = conflicts[0]
        self.assertEqual(conflict["conflict_type"], "cross_candidate_table_cell_mismatch")
        differences = {item["item_id"] for item in conflict["critical_differences"]}
        self.assertEqual(differences, {
            "table:row=23:value", "table:row=25:value", "table:row=28:value",
        })
        checklist = {item["id"]: item for item in conflict["verification_checklist"]}
        self.assertTrue(checklist["table:row=28:value"]["requires_double_read"])

    def test_dense_number_list_has_exhaustive_item_checklist(self):
        items = structured_math_items(
            "The first 10 values are 3, 5, 7, 11, 13, 17, 19, 23, 29, 31."
        )
        self.assertEqual(len(items), 10)
        self.assertEqual(items[0]["id"], "comma_list:item=0001")

    def test_literal_repeated_setup_is_factored_from_question(self):
        prefix = "The displayed numbers $5,19,101,619,4421,35899$ are each prime."
        setup = row("example:p10.x1", "example", prefix, 2)
        question = row(
            "conjecture:p10.q1", "conjecture",
            prefix + " Are there infinitely many such?", 2,
        )
        setup["section_id"] = question["section_id"] = "B43"
        events = factor_repeated_prefix_context([setup, question])
        self.assertEqual(len(events), 1)
        self.assertEqual(question["statement"], "Are there infinitely many such?")
        self.assertEqual(question["context_before_id"], setup["id"])


if __name__ == "__main__":
    unittest.main()
