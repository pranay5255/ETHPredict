from pathlib import Path

import yaml

from src.config.loader import ConfigManager, expand_grid_search, get_dotted


def test_v2_config_validates_and_loads_active_sections():
    manager = ConfigManager()
    cfg = manager.load_config("configs/config.yml")

    assert cfg.config_version == 2
    assert cfg.model_type.value == "lstm"
    assert cfg.targets["horizons"]["next_5m"]["bars"] == 1
    assert cfg.targets["horizons"]["next_hour"]["bars"] == 12
    assert cfg.validation["method"] == "purged_walk_forward"
    assert cfg.alpha_backtest["meta_threshold"] == 0.55
    assert cfg.tracking["trackio"]["project"] == "ethpredict"
    assert cfg.raw_config["tracking"]["trackio"]["auto_log_gpu"] is True


def test_legacy_config_is_preserved_with_deprecation_header():
    legacy = Path("configs/config_legacy.yml")
    assert legacy.exists()
    text = legacy.read_text(encoding="utf-8")
    assert text.startswith("# DEPRECATED: Legacy comprehensive ETHPredict configuration.")
    loaded = yaml.safe_load(text)
    assert "experiment" in loaded
    assert "market_maker" in loaded


def test_grid_expansion_uses_dotted_paths_and_max_trials():
    cfg = yaml.safe_load(Path("configs/config.yml").read_text(encoding="utf-8"))
    cfg["search"]["max_trials"] = 3

    trials = expand_grid_search(cfg)

    assert len(trials) == 3
    assert trials[0]["overrides"]["model.base.hidden_size"] == 16
    assert get_dotted(trials[0]["config"], "model.base.hidden_size") == 16
    assert get_dotted(trials[1]["config"], "alpha_backtest.edge_threshold_bps") in {5.0, 10.0}
