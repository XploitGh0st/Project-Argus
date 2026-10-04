import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    import os

    env = {"PYTHONPATH": str(ROOT / "src"), "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
    return subprocess.run(
        [sys.executable, "-m", "argus_cfg", *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_cli_on_mock_pickle(tmp_path: Path):
    pkl = ROOT / "fixtures" / "pkl" / "traffic_light.ast.pkl"
    proc = _run(str(pkl), "--out", str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    doc = json.loads((tmp_path / "cfg.json").read_text("utf-8"))
    assert doc["fsms"][0]["signal"] == "traffic.state"
    assert "FSMs       : 1" in proc.stdout


def test_cli_reports_bad_input(tmp_path: Path):
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"not a pickle")
    proc = _run(str(bad), "--out", str(tmp_path))
    assert proc.returncode != 0
