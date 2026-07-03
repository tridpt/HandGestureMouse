from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from app_config import AppConfig, copy_config_values, load_config, save_config


class LoadConfigTest(unittest.TestCase):
    def _write(self, payload: str) -> Path:
        tmp = Path(tempfile.mkdtemp()) / "settings.json"
        tmp.write_text(payload, encoding="utf-8")
        return tmp

    def test_missing_file_returns_defaults(self) -> None:
        missing = Path(tempfile.mkdtemp()) / "nope.json"
        self.assertEqual(load_config(missing), AppConfig())

    def test_valid_file_overrides_defaults(self) -> None:
        path = self._write(json.dumps({"smoothening": 8.0, "camera_index": 2}))
        config = load_config(path)
        self.assertEqual(config.smoothening, 8.0)
        self.assertEqual(config.camera_index, 2)
        # Untouched fields keep defaults.
        self.assertEqual(config.control_frame_margin, AppConfig().control_frame_margin)

    def test_unknown_keys_are_ignored(self) -> None:
        path = self._write(json.dumps({"smoothening": 6.0, "bogus_key": 123}))
        config = load_config(path)
        self.assertEqual(config.smoothening, 6.0)
        self.assertFalse(hasattr(config, "bogus_key"))

    def test_invalid_json_falls_back_to_defaults(self) -> None:
        path = self._write("{ this is not valid json ")
        self.assertEqual(load_config(path), AppConfig())

    def test_non_object_json_falls_back_to_defaults(self) -> None:
        path = self._write(json.dumps([1, 2, 3]))
        self.assertEqual(load_config(path), AppConfig())


class SaveConfigTest(unittest.TestCase):
    def test_save_then_load_roundtrip(self) -> None:
        path = Path(tempfile.mkdtemp()) / "settings.json"
        original = AppConfig(smoothening=7.5, left_click_distance_ratio=0.15)
        save_config(original, path)
        self.assertEqual(load_config(path), original)

    def test_saved_file_contains_all_fields(self) -> None:
        path = Path(tempfile.mkdtemp()) / "settings.json"
        save_config(AppConfig(), path)
        saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(set(saved), set(asdict(AppConfig())))


class CopyConfigValuesTest(unittest.TestCase):
    def test_copy_overwrites_every_field(self) -> None:
        target = AppConfig()
        source = AppConfig(smoothening=9.0, scroll_speed=1.2, camera_index=3)
        copy_config_values(target, source)
        self.assertEqual(target, source)
        # Same object, values mutated in place.
        self.assertIs(type(target), AppConfig)


if __name__ == "__main__":
    unittest.main()
