import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from parse_ocr import cli
from parse_ocr.common import resolve_config


class ResolveConfigTest(unittest.TestCase):
    def write(self, value: dict) -> Path:
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name) / "proj" / "run.json"
        path.parent.mkdir()
        path.write_text(json.dumps(value))
        return path

    def tearDown(self):
        if hasattr(self, "tmp"):
            self.tmp.cleanup()

    def test_relative_paths_resolve_against_config(self):
        path = self.write({"pdf": "../book.pdf", "output": "output", "profile": "book_profile.json"})
        config = resolve_config(path)
        self.assertEqual(Path(config["pdf"]), (path.parent.parent / "book.pdf").resolve())
        self.assertEqual(Path(config["output"]), (path.parent / "output").resolve())

    def test_base_url_expands_environment(self):
        path = self.write({"vlm": {"base_url": "${VLM_BASE_URL}", "model": "m"}})
        with mock.patch.dict(os.environ, {"VLM_BASE_URL": "http://localhost:9/v1"}):
            self.assertEqual(resolve_config(path)["vlm"]["base_url"], "http://localhost:9/v1")

    def test_unset_base_url_variable_fails_loudly(self):
        path = self.write({"vlm": {"base_url": "${PARSE_OCR_UNSET_TEST_VAR}", "model": "m"}})
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PARSE_OCR_UNSET_TEST_VAR", None)
            with self.assertRaises(RuntimeError):
                resolve_config(path)


class CliTest(unittest.TestCase):
    def test_every_command_module_imports(self):
        import importlib

        for module, _ in cli.COMMANDS.values():
            self.assertTrue(hasattr(importlib.import_module(f"parse_ocr.{module}"), "main"), module)

    def test_unknown_command_exits(self):
        with self.assertRaises(SystemExit):
            cli.main(["no-such-command"])


if __name__ == "__main__":
    unittest.main()
