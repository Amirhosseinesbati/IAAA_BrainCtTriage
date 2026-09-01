from __future__ import annotations

import json
import shlex
import tempfile
import unittest
from pathlib import Path

from scripts.launch_vast_mls_tmux import (
    _atomic_json,
    _resolve_manifest,
    _worker_command,
)


class VastMLSTmuxLauncherTests(unittest.TestCase):
    def _project(self, root: Path) -> Path:
        project = root / "project"
        (project / "config" / "experiments").mkdir(parents=True)
        return project

    def test_manifest_must_be_safe_mls_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = self._project(Path(temporary))
            manifest = project / "config" / "experiments" / "exp.yaml"
            manifest.write_text(
                "task: mls\nstrategy: mls_heatmap\nrun_name: mls-vast-exp14\n",
                encoding="utf-8",
            )
            resolved, run_name = _resolve_manifest(project, manifest)
            self.assertEqual(resolved, manifest.resolve())
            self.assertEqual(run_name, "mls-vast-exp14")

    def test_manifest_outside_experiment_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = self._project(root)
            manifest = root / "outside.yaml"
            manifest.write_text(
                "task: mls\nstrategy: mls_heatmap\nrun_name: run\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                _resolve_manifest(project, manifest)

    def test_worker_command_shell_quotes_paths(self) -> None:
        command = _worker_command(
            "/venv/bin/python",
            Path("/project with space/launcher.py"),
            Path("/project with space"),
            Path("/project with space/config/experiments/exp.yaml"),
            Path("/artifact root"),
            "mls_exp14",
        )
        tokens = shlex.split(command)
        self.assertTrue(tokens[1].endswith("launcher.py"))
        self.assertIn("project with space", tokens[1])
        self.assertIn("--worker", tokens)
        self.assertEqual(tokens[-1], "mls_exp14")

    def test_atomic_json_replaces_complete_document(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "status.json"
            _atomic_json(path, {"state": "running"})
            _atomic_json(path, {"state": "completed", "exit_code": 0})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"state": "completed", "exit_code": 0},
            )
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
