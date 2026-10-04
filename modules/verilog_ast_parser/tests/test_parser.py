import json
import subprocess
import sys
from pathlib import Path

import pytest

from argus_ast import ASTArtifact, TopModuleError, parse_verilog, parse_verilog_text


def test_parse_single_file(traffic_v: Path):
    art = parse_verilog(traffic_v)
    assert art.top_module == "traffic"
    assert art.module_names() == ["traffic"]
    assert type(art.ast).__name__ == "Source"


def test_parse_autodetects_top_in_multi_file(fixtures_dir: Path):
    mf = fixtures_dir / "multi_file"
    art = parse_verilog([mf / "top.v", mf / "alu.v"])
    assert art.top_module == "soc_top"
    assert sorted(art.module_names()) == ["alu", "soc_top"]


def test_explicit_top_validated(traffic_v: Path):
    with pytest.raises(TopModuleError, match="not defined"):
        parse_verilog(traffic_v, top_module="nope")


def test_rename_top(traffic_v: Path):
    art = parse_verilog(traffic_v, rename_top_to="top")
    assert art.top_module == "top"
    assert art.module_names() == ["top"]


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        parse_verilog(tmp_path / "missing.v")


def test_parse_text_with_comments():
    art = parse_verilog_text("// hi\nmodule m(input a, output y); assign y = a; endmodule")
    assert art.top_module == "m"


@pytest.mark.parametrize("name", ["counter.v", "fsm_traffic_light.v", "trojan_trigger.v"])
def test_all_fixtures_parse(fixtures_dir: Path, name: str):
    assert parse_verilog(fixtures_dir / name).module_names()


def test_save_and_load_roundtrip(traffic_v: Path, tmp_path: Path):
    art = parse_verilog(traffic_v)
    paths = art.save(tmp_path)
    assert paths["pickle"].is_file() and paths["json"].is_file()

    loaded = ASTArtifact.load(paths["pickle"])
    assert loaded.top_module == art.top_module
    assert loaded.module_names() == art.module_names()

    # lossless: Verilog regenerated from the reloaded AST is identical
    from pyverilog.ast_code_generator.codegen import ASTCodeGenerator

    gen = ASTCodeGenerator()
    assert gen.visit(loaded.ast) == gen.visit(art.ast)


def test_json_view_is_contract_shaped(traffic_v: Path, tmp_path: Path):
    paths = parse_verilog(traffic_v).save(tmp_path)
    doc = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert doc["schema_version"] == "1.0"
    assert doc["kind"] == "ast"
    assert doc["ast"]["type"] == "Source"
    assert doc["top_module"] == "traffic"


def test_load_rejects_wrong_kind(tmp_path: Path):
    import pickle

    bad = tmp_path / "bad.pkl"
    bad.write_bytes(pickle.dumps({"kind": "dfg", "schema_version": "1.0"}))
    with pytest.raises(ValueError, match="not an AST artifact"):
        ASTArtifact.load(bad)


def test_cli_end_to_end(traffic_v: Path, tmp_path: Path):
    src = Path(__file__).resolve().parent.parent / "src"
    proc = subprocess.run(
        [sys.executable, "-m", "argus_ast", str(traffic_v), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(src), "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "ast.pkl").is_file()
    assert "top module : traffic" in proc.stdout
