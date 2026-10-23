"""Unit tests for normalize_verilog."""
import pytest
from argus_ast.preprocess import normalize_verilog


def test_sized_literal_space():
    text = {"t.v": "module m; wire [33:0] x = 34'b 0011; endmodule"}
    norm, log = normalize_verilog(text)
    assert "34'b0011" in norm
    assert any("sized_literal" in l for l in log)


def test_include_expansion(tmp_path):
    inc = tmp_path / "inc.h"
    inc.write_text("parameter X = 1;")
    text = {"f.v": '`include "inc.h"\nmodule m; wire x = X; endmodule'}
    norm, log = normalize_verilog(text, [str(tmp_path)])
    assert "parameter X = 1;" in norm
    assert any("include" in l for l in log)


def test_define_expansion():
    text = {"t.v": "`define WIDTH 8\nmodule m; wire [`WIDTH-1:0] x; endmodule"}
    norm, log = normalize_verilog(text)
    assert "8-1:0" in norm or "[8-1:0]" in norm
    assert any("define" in l for l in log)


def test_ifdef():
    text = {"t.v": "`define FOO\n`ifdef FOO\nwire a;\n`else\nwire b;\n`endif\nmodule m; endmodule"}
    norm, log = normalize_verilog(text)
    assert "wire a;" in norm
    assert "wire b;" not in norm


def test_ifndef():
    text = {"t.v": "`ifndef BAR\nwire a;\n`else\nwire b;\n`endif\nmodule m; endmodule"}
    norm, log = normalize_verilog(text)
    assert "wire a;" in norm
    assert "wire b;" not in norm


def test_empty_port():
    text = {"t.v": "module m; wire x; sub s1(clk, , data); endmodule"}
    norm, log = normalize_verilog(text)
    assert ",," not in norm
    assert "__unconn_" in norm
    assert any("empty_port" in l for l in log)


def test_timescale_drop():
    text = {"t.v": "`timescale 1ns/1ps\nmodule m; endmodule"}
    norm, log = normalize_verilog(text)
    assert "`timescale" not in norm
    assert any("timescale" in l for l in log)


def test_define_func():
    text = {"t.v": "`define MAX(a,b) ((a)>(b)?(a):(b))\nmodule m; wire x = `MAX(3,4); endmodule"}
    norm, log = normalize_verilog(text)
    assert "((3)>(4)?(3):(4))" in norm
