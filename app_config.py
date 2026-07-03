from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path


SETTINGS_PATH = Path(__file__).with_name("settings.json")
LOGGER = logging.getLogger(__name__)


@dataclass
class AppConfig:
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    control_frame_margin: int = 90
    smoothening: float = 5.0
    safe_screen_margin: int = 4
    left_click_distance_ratio: float = 0.12
    left_click_release_ratio: float = 0.16
    right_click_distance_ratio: float = 0.10
    right_click_release_ratio: float = 0.14
    click_cooldown_seconds: float = 0.35
    drag_distance_ratio: float = 0.09
    drag_hold_seconds: float = 0.45
    scroll_spread_ratio: float = 0.18
    scroll_deadzone_px: int = 10
    scroll_speed: float = 0.35
    pause_fist_hold_seconds: float = 0.55
    pause_toggle_cooldown_seconds: float = 1.0


def load_config(path: Path = SETTINGS_PATH) -> AppConfig:
    defaults = AppConfig()
    if not path.exists():
        return defaults

    try:
        raw_config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        LOGGER.warning("Không đọc được settings.json, dùng cấu hình mặc định: %s", error)
        return defaults

    if not isinstance(raw_config, dict):
        LOGGER.warning("settings.json phải là object JSON, dùng cấu hình mặc định.")
        return defaults

    allowed_keys = {field.name for field in fields(AppConfig)}
    default_values = asdict(defaults)
    custom_values = {key: value for key, value in raw_config.items() if key in allowed_keys}
    unknown_keys = sorted(set(raw_config) - allowed_keys)
    if unknown_keys:
        LOGGER.warning("Bỏ qua cấu hình không hỗ trợ: %s", ", ".join(unknown_keys))

    try:
        return AppConfig(**{**default_values, **custom_values})
    except TypeError as error:
        LOGGER.warning("settings.json có giá trị không hợp lệ, dùng cấu hình mặc định: %s", error)
        return defaults


def save_config(config: AppConfig, path: Path = SETTINGS_PATH) -> None:
    text = json.dumps(asdict(config), ensure_ascii=False, indent=2)
    path.write_text(f"{text}\n", encoding="utf-8")


def copy_config_values(target: AppConfig, source: AppConfig) -> None:
    for field in fields(AppConfig):
        setattr(target, field.name, getattr(source, field.name))
