import json
from pathlib import Path

from argus_dfg.__main__ import main

HERE = Path(__file__).parent
EXPECTED_DIR = HERE.parent / "fixtures" / "expected"

def test_cli_success(tmp_path: Path, capsys) -> None:
    ast_file = EXPECTED_DIR / "no_fsm.ast.pkl"
    out_dir = tmp_path / "out"

    assert main([str(ast_file), "--out", str(out_dir)]) == 0

    stdout, stderr = capsys.readouterr()
    assert stderr == ""
    assert "top module : comb" in stdout
    assert "written    :" in stdout

    out_file = out_dir / "dfg.json"
    assert out_file.exists()
    
    with out_file.open() as fh:
        data = json.load(fh)
    
    assert data["top_module"] == "comb"
    assert data["kind"] == "dfg"

def test_cli_error(capsys) -> None:
    assert main(["non_existent.pkl"]) == 2
    stdout, stderr = capsys.readouterr()
    assert "error:" in stderr
