import os
import sys
from typer.testing import CliRunner

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from run import app


def test_help():
    runner = CliRunner()
    result = runner.invoke(app, ['--help'])
    assert result.exit_code == 0
    assert 'run' in result.output
