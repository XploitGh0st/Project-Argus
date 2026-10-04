"""Shared test configuration: fixture locations only (no cross-module imports)."""

from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "verilog"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture()
def traffic_v(fixtures_dir: Path) -> Path:
    return fixtures_dir / "fsm_traffic_light.v"
