import tempfile
import unittest
from pathlib import Path

import fitz


from parse_ocr.prose_claims import subpart_label, subpart_marker
from parse_ocr.text_spans import locate_span, plain_words, span_containment, span_from_excerpt, span_iou

TEXT = (
    "Let X be a random variable with finite variance. The moment generating function of X is defined "
    "as the expectation of exp(tX) for every real t. This definition can be extended to the "
    "essential supremum. Note that the triangle inequality fails for p below one."
)


class TextSpanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.pdf = Path(cls.tmp.name) / "sample.pdf"
        document = fitz.open()
        page = document.new_page()
        page.insert_textbox(fitz.Rect(50, 80, 550, 400), TEXT, fontsize=11)
        document.save(cls.pdf)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_plain_words_strip_math(self):
        self.assertEqual(plain_words("The *norm* $\\|X\\|_{L^p}$ is finite"), ["the", "norm", "is", "finite"])

    def test_locate_span_with_gap_in_anchor(self):
        # "$M_X(t)$" is stripped from the anchor, so the match must tolerate missing words.
        span = locate_span(self.pdf, [1], "The moment generating function of $X$", "for every real t.")
        self.assertTrue(span["found"])
        self.assertTrue(span["text"].startswith("The moment"))
        self.assertTrue(span["text"].rstrip(".").endswith("real t"))

    def test_iou_and_containment(self):
        whole = span_from_excerpt(self.pdf, [1], TEXT)
        part = span_from_excerpt(self.pdf, [1], "Note that the triangle inequality fails for p below one.")
        self.assertTrue(whole["found"] and part["found"])
        self.assertEqual(span_containment(part, whole), 1.0)
        self.assertLess(span_iou(part, whole), 0.5)

    def test_missing_anchor(self):
        self.assertFalse(locate_span(self.pdf, [1], "completely absent words here now", "")["found"])


class SubpartLabelTest(unittest.TestCase):
    def test_markers(self):
        self.assertEqual(subpart_marker("(b) Show that"), "b")
        self.assertEqual(subpart_marker("**(ii)** Then"), "ii")
        self.assertEqual(subpart_marker("Let (a) be"), "")

    def test_labels(self):
        self.assertEqual(subpart_label("3.13", {"source_excerpt": "(c) Prove"}), "3.13(c)")
        self.assertEqual(subpart_label("p4.e1", {"source_excerpt": "(c) Prove"}), "")


if __name__ == "__main__":
    unittest.main()


class SubpartCompletenessTest(unittest.TestCase):
    def test_missing_subparts_are_restored_verbatim(self):
        from parse_ocr.prose_claims import complete_subparts, split_subparts
        statement = "Minimize z for each system:\n\n(a)\n$$A$$\n\n(b)\n$$B$$\n\n(c)\n$$C$$"
        stem, segments = split_subparts(statement)
        self.assertEqual(stem, "Minimize z for each system:")
        self.assertEqual([marker for marker, _ in segments], ["a", "b", "c"])
        row = {"id": "exercise:32", "kind": "exercise", "statement": statement, "source_pages": [7]}
        parts = [{"uid": "u1", "part_index": 0, "source_excerpt": "(a)\n$$A$$", "normalized_statement": "N"}]
        restored = complete_subparts(row, parts)
        self.assertEqual(len(restored), 3)
        self.assertTrue(restored[1]["deterministic_restore"])
        self.assertEqual(restored[2]["normalized_statement"], "Minimize z for each system:\n\n(c)\n$$C$$")

    def test_unsplit_record_is_left_alone(self):
        from parse_ocr.prose_claims import complete_subparts
        row = {"id": "x", "kind": "exercise", "statement": "Show:\n(a) A\n(b) B"}
        parts = [{"uid": "u", "part_index": 0, "source_excerpt": "Show: (a) A (b) B", "normalized_statement": "N"}]
        self.assertEqual(complete_subparts(row, parts), parts)

    def test_inline_roman_numeral_is_not_a_subpart(self):
        from parse_ocr.prose_claims import split_subparts
        self.assertEqual(split_subparts("Let X be (i) bounded")[1], [])
