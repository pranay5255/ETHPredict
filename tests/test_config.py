import yaml


def test_config_valid():
    cfg = yaml.safe_load(open('configs/base.yml'))
    for key in ['experiment', 'bars', 'features', 'model']:
        assert key in cfg
