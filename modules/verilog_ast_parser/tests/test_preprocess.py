import pytest

from argus_ast import (
    TopModuleError,
    find_modules,
    find_top_module,
    rename_module,
    strip_comments,
)


def test_strip_line_and_block_comments_preserve_line_count():
    src = "a // c1\n/* x\ny */ b\nc"
    out = strip_comments(src)
    assert "c1" not in out and "x" not in out.replace("\n", "")
    assert out.count("\n") == src.count("\n")


def test_strip_comments_keeps_string_literals():
    src = 'initial $display("a // not a comment"); // real comment'
    out = strip_comments(src)
    assert '"a // not a comment"' in out
    assert "real comment" not in out


def test_legacy_bug_slash_in_expression_is_not_truncated():
    # the old implementation cut every line at the first '/', destroying division
    assert "a / b" in strip_comments("assign y = a / b;")


def test_find_modules_ignores_commented_modules():
    src = "// module ghost\nmodule real(input a);\nendmodule"
    assert find_modules(src) == ["real"]


def test_find_top_module_detects_uninstantiated_module():
    src = (
        "module alu(input a, output y); assign y = a; endmodule\n"
        "module soc(input a, output y); alu u(.a(a), .y(y)); endmodule\n"
    )
    assert find_top_module(src) == "soc"


def test_find_top_module_ambiguous_raises():
    src = "module a(); endmodule\nmodule b(); endmodule"
    with pytest.raises(TopModuleError, match="ambiguous"):
        find_top_module(src)


def test_find_top_module_none_raises():
    with pytest.raises(TopModuleError):
        find_top_module("wire x;")


def test_rename_uses_word_boundaries():
    src = "module alu(); endmodule module alu_core(); alu u(); endmodule"
    out = rename_module(src, "alu", "top")
    assert "module top()" in out
    assert "alu_core" in out  # not clobbered
