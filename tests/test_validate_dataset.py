from __future__ import annotations

import unittest
from pathlib import Path



from parse_ocr.validate_dataset import anaphoric_fragment, has_resolving_context  # noqa: E402


class ValidateDatasetTests(unittest.TestCase):
    def test_source_context_resolves_short_anaphoric_statement(self):
        row = {
            "statement": "Are any of them squares?",
            "context_before": "",
            "source_context": "Garcia found more than 200 harmonic numbers.",
            "source_context_host_id": "fact:host",
        }
        self.assertTrue(anaphoric_fragment(row["statement"]))
        self.assertTrue(has_resolving_context(row))

    def test_unlinked_anaphoric_statement_remains_unresolved(self):
        row = {"statement": "Are any of them squares?"}
        self.assertTrue(anaphoric_fragment(row["statement"]))
        self.assertFalse(has_resolving_context(row))
