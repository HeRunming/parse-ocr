import unittest
from pathlib import Path


from parse_ocr.direct_pipeline import is_summarized_table
from parse_ocr.prose_claims import keep_unmarked_exercise_whole, parse_verdicts, verify_prompt


class AddedAssumptionTest(unittest.TestCase):
    def test_listed_assumption_forces_unfaithful(self):
        raw = ("<audit><id>a</id><verdict>faithful</verdict><issues></issues>"
               "<added_assumptions>\"independent\"\n\"F_X:\\mathbb R\\to[0,1]\"</added_assumptions></audit>")
        row = parse_verdicts(raw)[0]
        self.assertEqual(row["verdict"], "unfaithful")
        self.assertEqual(len(row["added_assumptions"]), 2)
        self.assertEqual({issue["code"] for issue in row["issues"]}, {"added_hypothesis"})

    def test_none_and_missing_field_keep_verdict(self):
        raw = ("<audit><id>b</id><verdict>faithful</verdict><issues></issues><added_assumptions>None."
               "</added_assumptions></audit><audit><id>c</id><verdict>minor_issue</verdict><issues></issues></audit>")
        rows = {row["id"]: row for row in parse_verdicts(raw)}
        self.assertEqual(rows["b"]["verdict"], "faithful")
        self.assertEqual(rows["c"]["verdict"], "minor_issue")

    def test_book_context_is_rendered(self):
        self.assertIn("p3 definition 1.2", verify_prompt([], "- p3 definition 1.2: X is ..."))


class SummarizedTableTest(unittest.TestCase):
    def test_description_without_cells(self):
        row = {"statement": "The displayed table gives the successive tableaux for Cycle 0, Cycle 1.", "supporting_text": ""}
        self.assertTrue(is_summarized_table(row))

    def test_transcribed_table_and_plain_prose(self):
        self.assertFalse(is_summarized_table({"statement": "The displayed table gives\n| a | b |\n|---|---|\n| 1 | 2 |"}))
        self.assertFalse(is_summarized_table({"statement": "Let x be feasible. Then z is minimal."}))


class UnmarkedExerciseTest(unittest.TestCase):
    def test_parts_of_unmarked_exercise_are_merged(self):
        row = {"id": "exercise:30", "kind": "exercise",
               "statement": "In the system below ... prove an optimal solution must exist. Using Phase I and II construct one."}
        parts = [{"uid": "u1", "kind": "exercise", "source_excerpt": "prove ...", "normalized_statement": "A"},
                 {"uid": "u2", "kind": "exercise", "source_excerpt": "Using ...", "normalized_statement": "B"}]
        merged = keep_unmarked_exercise_whole(row, parts)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["component_uids"], ["u1", "u2"])
        self.assertEqual(merged[0]["source_excerpt"], row["statement"])

    def test_marked_exercise_and_non_exercise_untouched(self):
        marked = {"id": "e", "kind": "exercise", "statement": "Show:\n(a) A\n(b) B"}
        parts = [{"uid": "1", "kind": "exercise", "source_excerpt": "(a) A", "normalized_statement": "A"},
                 {"uid": "2", "kind": "exercise", "source_excerpt": "(b) B", "normalized_statement": "B"}]
        self.assertEqual(keep_unmarked_exercise_whole(marked, parts), parts)
        theorem = {"id": "t", "kind": "theorem", "statement": "A. B."}
        self.assertEqual(keep_unmarked_exercise_whole(theorem, parts), parts)


if __name__ == "__main__":
    unittest.main()
