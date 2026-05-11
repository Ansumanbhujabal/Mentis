from __future__ import annotations

from click.testing import CliRunner

from mentis.cli import cli


def test_help_lists_report_command() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "report" in result.output


def test_cache_clear_works(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    runner = CliRunner()
    result = runner.invoke(cli, ["cache", "clear"])
    assert result.exit_code == 0
