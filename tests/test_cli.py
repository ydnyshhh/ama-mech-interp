from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ama_mech_interp import cli


class CliTests(unittest.TestCase):
    def test_parser_includes_run_behavior_command(self) -> None:
        parser = cli.buildParser()
        subparser_action = next(
            action for action in parser._actions if getattr(action, "dest", None) == "command"
        )
        self.assertIn("run-behavior", subparser_action.choices)

    def test_minimum_viable_pipeline_prints_run_behavior_commands(self) -> None:
        fake_manifest = {
            "materialized_prompt_suites": SimpleNamespace(minimum_viable_prompt_count=40),
            "behavior_outputs": [{}, {}, {}, {}],
            "behavior_execution_commands": [
                "uv run ama-mech-interp run-behavior --manifest outputs/behavior/example/run_manifest.json",
            ],
        }
        output = io.StringIO()
        with patch("ama_mech_interp.cli.bootstrapMinimumViablePipeline", return_value=fake_manifest):
            with patch("sys.argv", ["ama-mech-interp", "minimum-viable-pipeline"]):
                with patch("sys.stdout", output):
                    cli.main()
        stdout = output.getvalue()
        self.assertIn("behavior_runs: 4", stdout)
        self.assertIn("run_behavior_command: uv run ama-mech-interp run-behavior --manifest", stdout)

    def test_run_behavior_manifest_path_dispatches_to_execution_runner(self) -> None:
        summary = SimpleNamespace(
            behavior_rows_path=Path("outputs/behavior/example/behavior_rows.jsonl"),
            manifest_path=Path("outputs/behavior/example/run_manifest.json"),
            prompt_count=40,
            executed_prompt_count=5,
            completed_prompt_count=5,
            failed_prompt_count=0,
            skipped_completed_count=35,
            remaining_planned_count=0,
        )
        output = io.StringIO()
        with patch("ama_mech_interp.cli.executeBehaviorRunFromManifest", return_value=summary) as execute_mock:
            with patch("sys.argv", ["ama-mech-interp", "run-behavior", "--manifest", "outputs/behavior/example/run_manifest.json"]):
                with patch("sys.stdout", output):
                    cli.main()
        execute_mock.assert_called_once()
        stdout = output.getvalue()
        self.assertIn("behavior_rows: outputs\\behavior\\example\\behavior_rows.jsonl", stdout)
        self.assertIn("executed_prompt_count: 5", stdout)


if __name__ == "__main__":
    unittest.main()
