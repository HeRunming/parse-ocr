from __future__ import annotations

import unittest
from pathlib import Path



from parse_ocr.extract_relationships import (  # noqa: E402
    build_source_context_memberships,
    dependency_prompt,
    explicit_cross_references,
    formalization_order,
    implicit_tasks,
    parse_dependencies,
    tokens,
)


def item(item_id: str, entry_id: str, section: str, kind: str, label: str, statement: str):
    return {
        "id": item_id,
        "entry_id": entry_id,
        "section_id": section,
        "kind": kind,
        "label": label,
        "statement": statement,
        "supporting_text": "",
        "dependencies": [],
    }


class RelationshipTests(unittest.TestCase):
    def test_dependency_evidence_token_gate(self):
        self.assertIn("x_k", tokens("q_k-X_k+1") & tokens("X_k is defined by a product"))
        self.assertFalse(tokens("the abundance is odd") & tokens("primitive abundant number"))

    def test_double_escaped_model_evidence_is_stored_as_exact_source_text(self):
        source = item(
            "fact:s", "A1.001", "A1", "fact", "s", "Smith's values are 6 & 21.",
        )
        target = item("fact:t", "A1.002", "A1", "fact", "t", "The values are 6 & 21.")
        raw = """<dependency>
<source_entry_id>A1.001</source_entry_id><target_entry_id>A1.002</target_entry_id>
<relation>continues</relation><evidence>Smith&amp;#x27;s values are 6 &amp;amp; 21.</evidence>
<target_evidence>The values are 6 &amp;amp; 21.</target_evidence><confidence>high</confidence>
</dependency>"""
        edges = parse_dependencies(raw, [source], [target])
        self.assertEqual(edges[0]["evidence"], "Smith's values are 6 & 21.")
        self.assertEqual(edges[0]["target_evidence"], "The values are 6 & 21.")

    def test_hybrid_source_context_groups_resolve_source_ids_to_entry_ids(self):
        rows = [item("definition:d1", "A1.001", "A1", "definition", "d1", "Define x.")]
        rows[0]["source_context_group_id"] = "src:p1.b1"
        groups = [{"group_id": "src:p1.b1", "source_ids": ["definition:d1"]}]
        self.assertEqual(
            build_source_context_memberships(rows, groups),
            [{"group_id": "src:p1.b1", "entry_id": "A1.001"}],
        )

    def test_explicit_section_and_typed_references(self):
        rows = [
            item("definition:1.1", "A1.001", "A1", "definition", "1.1", "A base definition."),
            item(
                "theorem:1.2", "A1.002", "A1", "theorem", "1.2",
                "Using Definition 1.1 and compare with B2; see A1; from Fig. 15(b).",
            ),
        ]
        sections = [{"section_id": "A1"}, {"section_id": "B2"}]
        refs = explicit_cross_references(rows, sections)["theorem:1.2"]
        targets = {(edge["target_type"], edge["target_label"], edge["resolved"]) for edge in refs}
        self.assertIn(("entry", "definition 1.1", True), targets)
        self.assertIn(("section", "B2", True), targets)
        self.assertIn(("section", "A1", True), targets)
        self.assertIn(("figure", "figure 15(b)", False), targets)

    def test_parenthesized_emphasized_and_range_section_references(self):
        rows = [
            item(
                "remark:r1", "C16.001", "C16", "remark", "r1",
                "Use the Carmichael results (A12, A13), the Legendre symbol "
                "(**F5**), ideas of C9 and E28, a result in A1, the first "
                "sentence of F13, and compare C14-16 with E10-14; see "
                r"\mathbf{{D9}}.",
            )
        ]
        section_ids = [
            "A1", "A12", "A13", "C9", "E28", "F5", "F13", "D9",
            "C14", "C15", "C16", "E10", "E11", "E12", "E13", "E14",
        ]
        refs = explicit_cross_references(
            rows, [{"section_id": section_id} for section_id in section_ids]
        )["remark:r1"]
        self.assertEqual(
            {edge["target_section_id"] for edge in refs}, set(section_ids)
        )

    def test_named_object_alias_resolves_to_page_label(self):
        rows = [
            item(
                "conjecture:p35.c1", "A8.001", "A8", "conjecture", "p35.c1",
                "Conjecture B of Hardy and Littlewood is that $P_k(n)$ has the stated asymptotic.",
            ),
            item(
                "fact:p36.f4", "A8.007", "A8", "fact", "p36.f4",
                "The count is larger than predicted by Conjecture B.",
            ),
        ]
        refs = explicit_cross_references(rows, [{"section_id": "A8"}])["fact:p36.f4"]
        match = next(edge for edge in refs if edge["target_label"] == "conjecture B")
        self.assertTrue(match["resolved"])
        self.assertEqual(match["target_entry_id"], "A8.001")

    def test_material_figure_reference_resolves_to_caption_object(self):
        rows = [
            item(
                "conjecture:q1", "D18.001", "D18", "conjecture", "q1",
                "Does a graph as in Figure 12 exist?",
            ),
            item(
                "remark:fig12", "D18.002", "D18", "remark", "fig12",
                "Figure 12. A graph of three cycles.",
            ),
        ]
        rows[1]["assets"] = [{"caption": "Figure 12. A graph of three cycles."}]
        refs = explicit_cross_references(rows, [{"section_id": "D18"}])["conjecture:q1"]
        match = next(edge for edge in refs if edge["target_label"] == "figure 12")
        self.assertTrue(match["resolved"])
        self.assertEqual(match["target_entry_id"], "D18.002")

    def test_parenthetical_bare_figure_reference_resolves_to_caption_object(self):
        rows = [
            item(
                "definition:q1", "A10.001", "A10", "definition", "q1",
                "Define the successive differences (Figure 2).",
            ),
            item(
                "example:fig2", "A10.002", "A10", "example", "fig2",
                "The displayed triangle.\n\nFigure 2. Successive differences.",
            ),
        ]
        refs = explicit_cross_references(rows, [{"section_id": "A10"}])["definition:q1"]
        match = next(edge for edge in refs if edge["target_label"] == "figure 2")
        self.assertTrue(match["resolved"])
        self.assertEqual(match["target_entry_id"], "A10.002")
        self.assertNotIn("example:fig2", explicit_cross_references(rows, [{"section_id": "A10"}]))

    def test_table_caption_in_title_is_indexed(self):
        rows = [
            item(
                "fact:q1", "C9.001", "C9", "fact", "q1",
                "The values are displayed in Table 3.",
            ),
            item(
                "example:table3", "C9.002", "C9", "example", "table3",
                "| k | value |\n|---|---|\n| 1 | 2 |",
            ),
        ]
        rows[1]["title"] = "Table 3. Values and exemplary sets."
        refs = explicit_cross_references(rows, [{"section_id": "C9"}])["fact:q1"]
        match = next(edge for edge in refs if edge["target_label"] == "table 3")
        self.assertTrue(match["resolved"])
        self.assertEqual(match["target_entry_id"], "C9.002")

    def test_possessive_external_table_is_not_a_local_coverage_gap(self):
        rows = [
            item(
                "fact:q1", "C10.001", "C10", "fact", "q1",
                "These values are from Baumert's Table 6.1.",
            )
        ]
        refs = explicit_cross_references(rows, [{"section_id": "C10"}])["fact:q1"]
        match = next(edge for edge in refs if edge["target_label"] == "table 6.1")
        self.assertTrue(match["resolved"])
        self.assertEqual(match["resolution_scope"], "external")
        self.assertEqual(match["external_owner"], "Baumert")

    def test_later_source_item_can_be_a_dependency(self):
        source = item("conjecture:q1", "A1.001", "A1", "conjecture", "q1", "Is every frob square?")
        target = item("definition:d1", "A1.002", "A1", "definition", "d1", "A frob is an integer of type F.")
        raw = """<dependency>
<source_entry_id>A1.001</source_entry_id>
<target_entry_id>A1.002</target_entry_id>
<relation>uses_definition</relation>
<evidence>frob</evidence>
<target_evidence>A frob</target_evidence>
<confidence>high</confidence>
</dependency>"""
        edges = parse_dependencies(raw, [source], [target])
        self.assertEqual(len(edges), 1)
        source["dependencies"] = edges
        order, cycles = formalization_order([source, target])
        self.assertEqual(order, ["A1.002", "A1.001"])
        self.assertEqual(cycles, [])

    def test_locally_defined_or_repeated_content_is_not_a_dependency(self):
        source = item(
            "conjecture:q1", "A1.001", "A1", "conjecture", "q1",
            "Let $M(x)$ be the number of special primes. Is $M(x)$ unbounded?",
        )
        target = item(
            "notation:d1", "A1.002", "A1", "notation", "d1",
            "Let $M(x)$ be the number of special primes.",
        )
        raw = """<dependency>
<source_entry_id>A1.001</source_entry_id>
<target_entry_id>A1.002</target_entry_id>
<relation>uses_notation</relation>
<evidence>Let $M(x)$ be the number of special primes.</evidence>
<target_evidence>Let $M(x)$ be the number of special primes.</target_evidence>
<confidence>high</confidence>
</dependency>"""
        self.assertEqual(parse_dependencies(raw, [source], [target]), [])

    def test_invalid_evidence_and_cycles_are_visible(self):
        source = item("theorem:t1", "A1.001", "A1", "theorem", "t1", "Every frob is finite.")
        target = item("definition:d1", "A1.002", "A1", "definition", "d1", "A frob is an integer.")
        raw = """<dependency>
<source_entry_id>A1.001</source_entry_id>
<target_entry_id>A1.002</target_entry_id>
<relation>uses_definition</relation>
<evidence>text not in the source</evidence>
<target_evidence>A frob</target_evidence>
<confidence>high</confidence>
</dependency>"""
        self.assertEqual(parse_dependencies(raw, [source], [target]), [])
        source["dependencies"] = [{"target_entry_id": "A1.002"}]
        target["dependencies"] = [{"target_entry_id": "A1.001"}]
        order, cycles = formalization_order([source, target])
        self.assertEqual(order, [])
        self.assertEqual(cycles, [["A1.001", "A1.002"]])

    def test_oversized_dependency_prompt_shards_candidates_without_dropping_them(self):
        rows = [
            item(
                f"fact:A1.{index}", f"A1.{index:03d}", "A1", "fact", str(index),
                f"Fact {index} uses symbol x_{index}. " + ("supporting prose " * 20),
            )
            for index in range(1, 11)
        ]
        config = {"relationships": {
            "source_batch_size": 5, "same_section_lookback": 5,
            "same_section_lookahead": 5, "cross_section_candidate_limit": 0,
        }}
        base = implicit_tasks(rows, {}, config)
        self.assertEqual(len(base), 2)
        mandatory_chars = len(dependency_prompt(base[0]["sources"], base[0]["sources"]))
        limit = (mandatory_chars + len(base[0]["prompt"])) // 2
        config["relationships"]["max_prompt_chars"] = limit
        sharded = implicit_tasks(rows, {}, config)
        first_batch = [task for task in sharded if task["id"].startswith("A1_0000")]
        self.assertGreater(len(first_batch), 1)
        self.assertTrue(all(len(task["prompt"]) <= limit for task in first_batch))
        covered = {row["entry_id"] for task in first_batch for row in task["candidates"]}
        expected = {row["entry_id"] for row in base[0]["candidates"]}
        self.assertEqual(covered, expected)


if __name__ == "__main__":
    unittest.main()
