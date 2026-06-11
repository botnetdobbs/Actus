"""
Tests for app/config.py: Settings and get_settings.
"""
from unittest.mock import patch


def test_scheduler_enabled_defaults_to_true():
    from app.config import get_settings
    get_settings.cache_clear()
    try:
        s = get_settings()
        assert s.scheduler_enabled is True
    finally:
        get_settings.cache_clear()


def test_scheduler_enabled_can_be_disabled():
    from app.config import get_settings, Settings
    get_settings.cache_clear()
    try:
        with patch.dict("os.environ", {"SCHEDULER_ENABLED": "false", "DEBUG": "true"}):
            s = Settings()
            assert s.scheduler_enabled is False
    finally:
        get_settings.cache_clear()
