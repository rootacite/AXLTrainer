"""install_daemon.py without Gradle, systemctl, or `--serve`."""

import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import install_daemon


class InstallDaemonTest(unittest.TestCase):
    def test_unit_text_names_both_children(self):
        repo = Path("/home/axl/AxlTrainer")
        web = Path("/home/axl/.local/share/axlranko/web")
        python = "/opt/conda/envs/axl/bin/python"
        text = install_daemon.render_unit(python, repo, web)
        api_cmd, web_cmd = install_daemon.child_commands(python, repo, web)

        self.assertIn(install_daemon.command_line(api_cmd), text)
        self.assertIn(install_daemon.command_line(web_cmd), text)
        self.assertIn("--bind 0.0.0.0", text)
        self.assertIn(str(web), text)
        self.assertIn("192.168.0.0/16", text)
        self.assertIn("--host 0.0.0.0", text)
        self.assertIn("install_daemon.py --serve", text)
        self.assertIn(f"WorkingDirectory={repo}", text)
        self.assertIn(f"Environment=AXL_PYTHON={python}", text)
        # The unit starts the supervisor; it does not exec the children itself.
        self.assertIn(f"ExecStart={python} -u {repo / 'install_daemon.py'} --serve", text)
        self.assertNotIn("ExecStart=" + install_daemon.command_line(api_cmd), text)

    def test_install_web_tree_replaces_the_previous_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src"
            dest = root / "dest"
            (src / "nested").mkdir(parents=True)
            (src / "index.html").write_text("new", encoding="utf-8")
            (src / "nested" / "app.js").write_text("js", encoding="utf-8")
            dest.mkdir()
            (dest / "old.txt").write_text("gone", encoding="utf-8")

            install_daemon.install_web_tree(src, dest)

            self.assertEqual((dest / "index.html").read_text(encoding="utf-8"), "new")
            self.assertEqual((dest / "nested" / "app.js").read_text(encoding="utf-8"), "js")
            self.assertFalse((dest / "old.txt").exists())

    def test_axl_python_override_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            python = Path(tmp) / "python"
            python.write_text("#!/bin/sh\n", encoding="utf-8")
            python.chmod(python.stat().st_mode | stat.S_IEXEC)
            found = install_daemon.resolve_python(
                {"AXL_PYTHON": str(python)},
                home=Path(tmp) / "no-home",
                opt_conda=Path(tmp) / "no-opt",
            )
            self.assertEqual(found, str(python))

    def test_conda_env_python_is_found_from_conda_exe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            conda = root / "miniconda3" / "bin" / "conda"
            conda.parent.mkdir(parents=True)
            conda.write_text("", encoding="utf-8")
            python = root / "miniconda3" / "envs" / "axl" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("", encoding="utf-8")
            found = install_daemon.resolve_python(
                {"CONDA_EXE": str(conda)},
                home=root / "no-home",
                opt_conda=root / "no-opt",
            )
            self.assertEqual(found, str(python))

    def test_current_axl_env_uses_this_interpreter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            found = install_daemon.resolve_python(
                {"CONDA_DEFAULT_ENV": "axl"},
                home=root / "no-home",
                opt_conda=root / "no-opt",
            )
            self.assertEqual(found, sys.executable)

    def test_missing_interpreter_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(SystemExit):
                install_daemon.resolve_python(
                    {},
                    home=root / "no-home",
                    opt_conda=root / "no-opt",
                )

    def test_missing_axl_python_override_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                install_daemon.resolve_python(
                    {"AXL_PYTHON": str(Path(tmp) / "missing")},
                    home=Path(tmp) / "no-home",
                    opt_conda=Path(tmp) / "no-opt",
                )


if __name__ == "__main__":
    unittest.main()
