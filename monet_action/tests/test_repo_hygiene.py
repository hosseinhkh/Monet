"""Repository hygiene: everything needed on the GPU server is tracked by git and loads.

(The upstream root .gitignore ignores every directory named ``scripts/``; this test guards the
override in monet_action/.gitignore.)
"""
import shutil
import subprocess

import pytest

from monet_action.config import load_config

from conftest import ROOT

ESSENTIAL = ["scripts/setup_env.sh", "scripts/train_d1.sh", "scripts/evaluate_all.sh", "scripts/run_phase1.sh",
             "configs/base.yaml", "configs/d2_lora.yaml", "src/monet_action/train.py", "environment/train_requirements.txt",
             "Makefile", "README.md"]


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
@pytest.mark.parametrize("rel", ESSENTIAL)
def test_essential_files_not_git_ignored(rel):
    p = ROOT / rel
    assert p.exists(), rel
    r = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", str(p)], capture_output=True)
    assert r.returncode == 1, f"{rel} is ignored by a .gitignore rule"


def test_scripts_are_executable_and_strict():
    for sh in sorted((ROOT / "scripts").glob("*.sh")):
        assert sh.stat().st_mode & 0o111, f"{sh.name} is not executable"
        text = sh.read_text()
        assert text.startswith("#!/usr/bin/env bash"), sh.name
        # setup_env.sh uses `set -eo pipefail` and enables -u only after `conda activate`
        # (conda's activation scripts reference unset variables)
        assert "common.sh" in text or "set -euo pipefail" in text or "set -eo pipefail" in text, f"{sh.name}: no strict mode"
        last = [ln for ln in text.strip().splitlines() if ln.strip() and not ln.lstrip().startswith("#")][-1]
        assert not last.lstrip().startswith("["), f"{sh.name}: a final `[ ... ] && ...` makes the script exit 1"


@pytest.mark.parametrize("cfg", sorted(p.name for p in (ROOT / "configs").glob("*.yaml")))
def test_every_config_loads_and_validates(cfg):
    load_config(str(ROOT / "configs" / cfg))


def test_no_debugger_leftovers_in_package():
    for py in (ROOT / "src").rglob("*.py"):
        t = py.read_text()
        assert "import pdb" not in t and "breakpoint()" not in t, py
