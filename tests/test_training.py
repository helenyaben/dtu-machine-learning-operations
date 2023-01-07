
from click.testing import CliRunner
from src.models import train_model

def test_training():
  runner = CliRunner()
  result = runner.invoke(train_model.train, ['--lr', '1e-3'])
  assert result.exit_code == 0, 'Error in training execution'
#   assert 'Training day and night' in result.output, ''
