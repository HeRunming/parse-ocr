import unittest
from pathlib import Path


from parse_ocr.prose_claims import dedupe_claims, parse_claims, parse_normalized, parse_verdicts, token_containment


class ProseClaimParsingTest(unittest.TestCase):
    def test_parse_claim_maps_slots_to_pages(self):
        raw = """<claim><kind>observation</kind><name>tail formula</name><start_slot>2</start_slot>
<end_slot>2</end_slot><anchor>Note that</anchor><source_excerpt>Note that $E X=\\int_0^\\infty P\\{X>t\\}dt$.</source_excerpt>
<context_excerpts></context_excerpts><normalized_statement>Let $X\\ge 0$. Then ...</normalized_statement>
<assertion_status>bogus</assertion_status><justification_excerpt></justification_excerpt><open_issues></open_issues></claim>"""
        rows = parse_claims(raw, {}, [10, 11])
        self.assertEqual(rows[0]["kind"], "fact")
        self.assertEqual(rows[0]["source_pages"], [11])
        self.assertEqual(rows[0]["assertion_status"], "stated_without_proof")

    def test_parse_claim_rejects_unparseable(self):
        with self.assertRaises(ValueError):
            parse_claims("garbage", {}, [1])
        self.assertEqual(parse_claims("<no_claims/>", {}, [1]), [])

    def test_parse_normalized_parts(self):
        raw = """<normalized><id>definition:p3.d1</id>
<part><kind>definition</kind><source_excerpt>A</source_excerpt><normalized_statement>S1</normalized_statement><open_issues></open_issues></part>
<part><kind>fact</kind><source_excerpt>B</source_excerpt><normalized_statement>S2</normalized_statement><open_issues>x</open_issues></part>
</normalized>"""
        rows = parse_normalized(raw)
        self.assertEqual([part["normalized_statement"] for part in rows[0]["parts"]], ["S1", "S2"])

    def test_parse_verdict_uses_audit_wrapper(self):
        raw = "<audit><id>pc_1</id><verdict>unfaithful</verdict><issues>weakened: lost bound\nfoo: bar</issues></audit>"
        rows = parse_verdicts(raw)
        self.assertEqual(rows[0]["verdict"], "unfaithful")
        self.assertEqual([issue["code"] for issue in rows[0]["issues"]], ["weakened", "other"])

    def test_dedupe_keeps_longest_overlapping_window_copy(self):
        base = {"kind": "fact", "source_pages": [4], "uid": "a",
                "source_excerpt": "the variance of the sample mean shrinks to zero as N grows",
                "normalized_statement": "short"}
        longer = dict(base, uid="b", normalized_statement="a longer normalized statement")
        other_page = dict(base, uid="c", source_pages=[9])
        kept = dedupe_claims([base, longer, other_page])
        self.assertEqual(sorted(item["uid"] for item in kept), ["b", "c"])

    def test_token_containment(self):
        self.assertEqual(token_containment(["a", "b"], ["a", "b", "c"]), 1.0)
        self.assertEqual(token_containment([], ["a"]), 0.0)


if __name__ == "__main__":
    unittest.main()
