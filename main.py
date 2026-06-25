from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

from app_config import AppConfig, SETTINGS_PATH, copy_config_values, load_config, save_config
from gesture_logic import analyze_hand, is_fist_gesture, is_scroll_gesture, landmark_to_pixel

try:
    import msvcrt
except ImportError:  # pragma: no cover - Windows-only convenience.
    msvcrt = None


MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
WINDOW_NAME = "Hand Gesture Mouse"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

LOGGER = logging.getLogger(__name__)


@dataclass
class MouseState:
    previous_x: float = 0.0
    previous_y: float = 0.0
    left_click_active: bool = False
    right_click_active: bool = False
    drag_started_at: float | None = None
    is_dragging: bool = False
    scroll_anchor_y: float | None = None
    control_enabled: bool = True
    fist_started_at: float | None = None
    pause_gesture_armed: bool = True
    last_pause_toggle_at: float = 0.0
    status: str = "Move"
    status_until: float = 0.0


@dataclass
class UiState:
    preview_hidden: bool = False
    config_dirty: bool = False
    notice: str = ""
    notice_until: float = 0.0


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def build_landmarker_options(model_path: Path) -> HandLandmarkerOptions:
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

    return HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        num_hands=1,
        running_mode=VisionRunningMode.IMAGE,
    )


def open_camera(config: AppConfig) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(config.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Không mở được camera. Hãy kiểm tra quyền camera hoặc camera index.")

    return cap


def map_camera_to_screen(
    x: int,
    y: int,
    image_width: int,
    image_height: int,
    screen_width: int,
    screen_height: int,
    config: AppConfig,
) -> tuple[float, float]:
    left = config.control_frame_margin
    right = max(left + 1, image_width - config.control_frame_margin)
    top = config.control_frame_margin
    bottom = max(top + 1, image_height - config.control_frame_margin)

    clamped_x = float(np.clip(x, left, right))
    clamped_y = float(np.clip(y, top, bottom))

    screen_x = np.interp(
        clamped_x,
        (left, right),
        (
            config.safe_screen_margin,
            max(config.safe_screen_margin, screen_width - config.safe_screen_margin),
        ),
    )
    screen_y = np.interp(
        clamped_y,
        (top, bottom),
        (
            config.safe_screen_margin,
            max(config.safe_screen_margin, screen_height - config.safe_screen_margin),
        ),
    )
    return float(screen_x), float(screen_y)


def set_status(state: MouseState, status: str, hold_seconds: float = 0.0) -> None:
    state.status = status
    state.status_until = time.monotonic() + hold_seconds if hold_seconds else 0.0


def current_status(state: MouseState) -> str:
    now = time.monotonic()
    if state.status_until and now > state.status_until:
        state.status_until = 0.0
        state.status = "Paused" if not state.control_enabled else "Move"

    if not state.control_enabled and not state.status_until:
        return "Paused"

    return state.status


def set_notice(ui_state: UiState, message: str, hold_seconds: float = 1.5) -> None:
    ui_state.notice = message
    ui_state.notice_until = time.monotonic() + hold_seconds
    LOGGER.info(message)


def current_notice(ui_state: UiState) -> str:
    if ui_state.notice_until and time.monotonic() > ui_state.notice_until:
        ui_state.notice = ""
        ui_state.notice_until = 0.0
    return ui_state.notice


def move_mouse(x: float, y: float) -> bool:
    try:
        pyautogui.moveTo(int(x), int(y), duration=0)
    except pyautogui.FailSafeException:
        LOGGER.warning("Đã kích hoạt PyAutoGUI fail-safe. Dừng chương trình.")
        return False
    except Exception as error:
        LOGGER.warning("Không thể di chuyển chuột. Dừng chương trình: %s", error)
        return False
    return True


def click_mouse(button: str = "left") -> bool:
    try:
        pyautogui.click(button=button)
    except pyautogui.FailSafeException:
        LOGGER.warning("Đã kích hoạt PyAutoGUI fail-safe. Dừng chương trình.")
        return False
    except Exception as error:
        LOGGER.warning("Không thể click chuột. Dừng chương trình: %s", error)
        return False
    return True


def mouse_down() -> bool:
    try:
        pyautogui.mouseDown()
    except pyautogui.FailSafeException:
        LOGGER.warning("Đã kích hoạt PyAutoGUI fail-safe. Dừng chương trình.")
        return False
    except Exception as error:
        LOGGER.warning("Không thể giữ chuột. Dừng chương trình: %s", error)
        return False
    return True


def mouse_up() -> bool:
    try:
        pyautogui.mouseUp()
    except pyautogui.FailSafeException:
        LOGGER.warning("Đã kích hoạt PyAutoGUI fail-safe. Dừng chương trình.")
        return False
    except Exception as error:
        LOGGER.warning("Không thể thả chuột. Dừng chương trình: %s", error)
        return False
    return True


def scroll_mouse(clicks: int) -> bool:
    try:
        pyautogui.scroll(clicks)
    except pyautogui.FailSafeException:
        LOGGER.warning("Đã kích hoạt PyAutoGUI fail-safe. Dừng chương trình.")
        return False
    except Exception as error:
        LOGGER.warning("Không thể scroll. Dừng chương trình: %s", error)
        return False
    return True


def release_drag(state: MouseState) -> bool:
    state.drag_started_at = None
    if not state.is_dragging:
        return True

    state.is_dragging = False
    set_status(state, "Drop", hold_seconds=0.4)
    return mouse_up()


def reset_gestures(state: MouseState) -> None:
    state.left_click_active = False
    state.right_click_active = False
    state.scroll_anchor_y = None


def reset_pause_gesture(state: MouseState) -> None:
    state.fist_started_at = None
    state.pause_gesture_armed = True


def handle_pause_gesture(state: MouseState, config: AppConfig) -> bool:
    now = time.monotonic()
    if state.fist_started_at is None:
        state.fist_started_at = now

    held_long_enough = now - state.fist_started_at >= config.pause_fist_hold_seconds
    cooled_down = now - state.last_pause_toggle_at >= config.pause_toggle_cooldown_seconds

    if state.pause_gesture_armed and held_long_enough and cooled_down:
        state.control_enabled = not state.control_enabled
        state.last_pause_toggle_at = now
        state.pause_gesture_armed = False
        status = "Paused" if not state.control_enabled else "Control on"
        set_status(state, status, hold_seconds=0.9)
        return True

    set_status(state, "Hold fist" if state.control_enabled else "Hold fist to resume")
    return True


def draw_landmarks(image: Any, hand_landmarks: list[Any]) -> None:
    height, width, _ = image.shape
    for lm in hand_landmarks:
        cx, cy = landmark_to_pixel(lm, width, height)
        cv2.circle(image, (cx, cy), 4, (0, 255, 0), cv2.FILLED)


def draw_control_frame(image: Any, config: AppConfig) -> None:
    height, width, _ = image.shape
    cv2.rectangle(
        image,
        (config.control_frame_margin, config.control_frame_margin),
        (width - config.control_frame_margin, height - config.control_frame_margin),
        (255, 0, 255),
        2,
    )


def draw_status(image: Any, state: MouseState, config: AppConfig, ui_state: UiState) -> None:
    dirty_marker = "*" if ui_state.config_dirty else ""
    cv2.rectangle(image, (14, 68), (376, 136), (20, 20, 20), cv2.FILLED)
    cv2.putText(
        image,
        f"Mode: {current_status(state)}",
        (24, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.74,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        image,
        (
            f"Smooth {config.smoothening:.1f}  "
            f"Click {config.left_click_distance_ratio:.2f}  "
            f"Frame {config.control_frame_margin}{dirty_marker}"
        ),
        (24, 122),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (210, 230, 255),
        1,
    )


def draw_notice(image: Any, ui_state: UiState) -> None:
    notice = current_notice(ui_state)
    if not notice:
        return

    height, width, _ = image.shape
    cv2.rectangle(image, (14, height - 48), (width - 14, height - 14), (30, 30, 30), cv2.FILLED)
    cv2.putText(
        image,
        notice[:74],
        (24, height - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )


def draw_help_panel(image: Any, state: MouseState) -> None:
    height, width, _ = image.shape
    panel_width = 238
    x0 = max(10, width - panel_width - 10)
    y0 = 14
    x1 = width - 10
    y1 = min(height - 58, y0 + 340)
    cv2.rectangle(image, (x0, y0), (x1, y1), (18, 18, 18), cv2.FILLED)
    cv2.rectangle(image, (x0, y0), (x1, y1), (80, 80, 80), 1)

    lines = [
        ("Gestures", (0, 255, 255)),
        ("Move: index finger", (245, 245, 245)),
        ("Left: index + middle", (245, 245, 245)),
        ("Right: middle + ring", (245, 245, 245)),
        ("Drag: thumb + index", (245, 245, 245)),
        ("Scroll: V sign up/down", (245, 245, 245)),
        ("Pause: hold fist", (245, 245, 245)),
        ("", (245, 245, 245)),
        ("Hotkeys", (0, 255, 255)),
        ("+/- smooth", (215, 230, 255)),
        ("[ ] click threshold", (215, 230, 255)),
        (", . control frame", (215, 230, 255)),
        ("S save   M background", (215, 230, 255)),
        ("P pause  Q quit", (215, 230, 255)),
    ]

    y = y0 + 28
    for text, color in lines:
        if text:
            cv2.putText(
                image,
                text,
                (x0 + 12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
            )
        y += 23

    status = "ON" if state.control_enabled else "PAUSED"
    color = (80, 220, 120) if state.control_enabled else (0, 165, 255)
    cv2.putText(
        image,
        f"Control: {status}",
        (x0 + 12, y1 - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
    )


def draw_fps(image: Any, previous_time: float) -> float:
    current_time = time.time()
    fps = 1 / (current_time - previous_time) if current_time > previous_time else 0
    cv2.putText(
        image,
        f"FPS: {int(fps)}",
        (20, 50),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        (255, 0, 0),
        3,
    )
    return current_time


def handle_scroll(
    image: Any,
    state: MouseState,
    config: AppConfig,
    index_tip: tuple[int, int],
    middle_tip: tuple[int, int],
) -> bool:
    midpoint_y = (index_tip[1] + middle_tip[1]) / 2
    if state.scroll_anchor_y is None:
        state.scroll_anchor_y = midpoint_y
        set_status(state, "Scroll")
        return True

    delta_y = midpoint_y - state.scroll_anchor_y
    if abs(delta_y) < config.scroll_deadzone_px:
        set_status(state, "Scroll")
        return True

    clicks = int(-delta_y * config.scroll_speed)
    if clicks == 0:
        clicks = -1 if delta_y > 0 else 1

    cv2.line(image, index_tip, middle_tip, (0, 255, 255), 3)
    set_status(state, "Scroll up" if clicks > 0 else "Scroll down")
    state.scroll_anchor_y = midpoint_y
    return scroll_mouse(clicks)


def handle_drag(
    state: MouseState,
    config: AppConfig,
    thumb_index_ratio: float,
) -> bool:
    if thumb_index_ratio >= config.drag_distance_ratio:
        return release_drag(state)

    state.left_click_active = False
    state.right_click_active = False
    state.scroll_anchor_y = None

    now = time.monotonic()
    if state.drag_started_at is None:
        state.drag_started_at = now
        set_status(state, "Hold drag")
        return True

    if not state.is_dragging and now - state.drag_started_at >= config.drag_hold_seconds:
        if not mouse_down():
            return False
        state.is_dragging = True

    set_status(state, "Dragging" if state.is_dragging else "Hold drag")
    return True


def handle_clicks(
    state: MouseState,
    index_middle_ratio: float,
    middle_ring_ratio: float,
    config: AppConfig,
) -> bool:
    if index_middle_ratio < config.left_click_distance_ratio:
        state.right_click_active = False
        state.scroll_anchor_y = None
        if not state.left_click_active:
            if not click_mouse("left"):
                return False
            state.left_click_active = True
            set_status(state, "Left click", hold_seconds=0.4)
        return True

    state.left_click_active = False

    if middle_ring_ratio < config.right_click_distance_ratio:
        state.scroll_anchor_y = None
        if not state.right_click_active:
            if not click_mouse("right"):
                return False
            state.right_click_active = True
            set_status(state, "Right click", hold_seconds=0.4)
        return True

    state.right_click_active = False
    return True


def handle_hand(
    image: Any,
    hand_landmarks: list[Any],
    state: MouseState,
    screen_size: tuple[int, int],
    config: AppConfig,
) -> bool:
    if len(hand_landmarks) <= 20:
        reset_gestures(state)
        reset_pause_gesture(state)
        return release_drag(state)

    height, width, _ = image.shape
    screen_width, screen_height = screen_size
    metrics = analyze_hand(hand_landmarks, width, height)

    draw_landmarks(image, hand_landmarks)
    cv2.circle(image, metrics.index_tip, 15, (255, 0, 255), cv2.FILLED)
    cv2.circle(image, metrics.middle_tip, 12, (255, 0, 255), cv2.FILLED)
    cv2.circle(image, metrics.ring_tip, 10, (255, 0, 255), cv2.FILLED)

    if is_fist_gesture(metrics):
        reset_gestures(state)
        if not release_drag(state):
            return False
        return handle_pause_gesture(state, config)

    reset_pause_gesture(state)
    if not state.control_enabled:
        reset_gestures(state)
        set_status(state, "Paused")
        return release_drag(state)

    target_x, target_y = map_camera_to_screen(
        metrics.index_tip[0],
        metrics.index_tip[1],
        width,
        height,
        screen_width,
        screen_height,
        config,
    )

    current_x = state.previous_x + (target_x - state.previous_x) / config.smoothening
    current_y = state.previous_y + (target_y - state.previous_y) / config.smoothening

    if not move_mouse(current_x, current_y):
        return False

    state.previous_x = current_x
    state.previous_y = current_y

    if metrics.thumb_index_ratio < config.drag_distance_ratio or state.drag_started_at is not None:
        cv2.line(image, metrics.thumb_tip, metrics.index_tip, (0, 165, 255), 3)
        return handle_drag(state, config, metrics.thumb_index_ratio)

    if not handle_clicks(
        state,
        metrics.index_middle_ratio,
        metrics.middle_ring_ratio,
        config,
    ):
        return False

    if state.left_click_active or state.right_click_active:
        return True

    if is_scroll_gesture(metrics, config):
        return handle_scroll(image, state, config, metrics.index_tip, metrics.middle_tip)

    state.scroll_anchor_y = None
    if not state.status_until:
        set_status(state, "Move")
    return True


def normalize_key(raw_key: int) -> str | None:
    if raw_key in (-1, 255):
        return None
    if raw_key == 27:
        return "escape"
    try:
        return chr(raw_key).lower()
    except ValueError:
        return None


def read_cv2_key() -> str | None:
    return normalize_key(cv2.waitKey(1) & 0xFF)


def read_terminal_key() -> str | None:
    if msvcrt is None or not msvcrt.kbhit():
        return None

    key = msvcrt.getwch()
    if key in ("\x00", "\xe0"):
        if msvcrt.kbhit():
            msvcrt.getwch()
        return None
    if key == "\x1b":
        return "escape"
    return key.lower()


def safe_destroy_window() -> None:
    try:
        cv2.destroyWindow(WINDOW_NAME)
    except cv2.error:
        pass


def adjust_float(
    config: AppConfig,
    field_name: str,
    delta: float,
    minimum: float,
    maximum: float,
    digits: int,
    ui_state: UiState,
    label: str,
) -> None:
    current_value = float(getattr(config, field_name))
    new_value = round(min(max(current_value + delta, minimum), maximum), digits)
    setattr(config, field_name, new_value)
    ui_state.config_dirty = True
    set_notice(ui_state, f"{label}: {new_value} (press S to save)")


def adjust_int(
    config: AppConfig,
    field_name: str,
    delta: int,
    minimum: int,
    maximum: int,
    ui_state: UiState,
    label: str,
) -> None:
    current_value = int(getattr(config, field_name))
    new_value = min(max(current_value + delta, minimum), maximum)
    setattr(config, field_name, new_value)
    ui_state.config_dirty = True
    set_notice(ui_state, f"{label}: {new_value} (press S to save)")


def toggle_manual_pause(state: MouseState) -> None:
    state.control_enabled = not state.control_enabled
    reset_gestures(state)
    status = "Paused" if not state.control_enabled else "Control on"
    set_status(state, status, hold_seconds=0.8)


def handle_hotkey(
    key: str | None,
    config: AppConfig,
    mouse_state: MouseState,
    ui_state: UiState,
) -> bool:
    if key is None:
        return True

    if key in ("q", "escape"):
        return False

    if key == "m":
        ui_state.preview_hidden = not ui_state.preview_hidden
        if ui_state.preview_hidden:
            safe_destroy_window()
            set_notice(ui_state, "Background mode: press M in terminal to show preview")
        else:
            set_notice(ui_state, "Preview shown")
        return True

    if key == "p":
        toggle_manual_pause(mouse_state)
        set_notice(ui_state, "Manual pause toggled")
        return True

    if key == "s":
        save_config(config)
        ui_state.config_dirty = False
        set_notice(ui_state, f"Saved {SETTINGS_PATH.name}")
        return True

    if key == "r":
        copy_config_values(config, load_config())
        ui_state.config_dirty = False
        set_notice(ui_state, f"Reloaded {SETTINGS_PATH.name}")
        return True

    if key in ("+", "="):
        adjust_float(config, "smoothening", 0.5, 1.0, 15.0, 1, ui_state, "Smoothening")
        return True

    if key in ("-", "_"):
        adjust_float(config, "smoothening", -0.5, 1.0, 15.0, 1, ui_state, "Smoothening")
        return True

    if key == "[":
        adjust_float(
            config,
            "left_click_distance_ratio",
            -0.01,
            0.05,
            0.25,
            2,
            ui_state,
            "Left click threshold",
        )
        return True

    if key == "]":
        adjust_float(
            config,
            "left_click_distance_ratio",
            0.01,
            0.05,
            0.25,
            2,
            ui_state,
            "Left click threshold",
        )
        return True

    if key == ",":
        adjust_int(config, "control_frame_margin", -5, 20, 180, ui_state, "Control frame")
        return True

    if key == ".":
        adjust_int(config, "control_frame_margin", 5, 20, 180, ui_state, "Control frame")
        return True

    if key == "9":
        adjust_float(config, "scroll_speed", -0.05, 0.05, 1.5, 2, ui_state, "Scroll speed")
        return True

    if key == "0":
        adjust_float(config, "scroll_speed", 0.05, 0.05, 1.5, 2, ui_state, "Scroll speed")
        return True

    return True


def process_keys(config: AppConfig, mouse_state: MouseState, ui_state: UiState) -> bool:
    terminal_key = read_terminal_key()
    if not handle_hotkey(terminal_key, config, mouse_state, ui_state):
        return False
    return True


def run() -> int:
    configure_logging()

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0

    config = load_config()
    options = build_landmarker_options(MODEL_PATH)
    screen_size = tuple(pyautogui.size())
    mouse_state = MouseState()
    ui_state = UiState()
    previous_time = time.time()
    cap = open_camera(config)

    LOGGER.info("Đang tải AI model nhận diện tay và mở camera...")
    LOGGER.info("Đang dùng cấu hình: %s", SETTINGS_PATH.name if SETTINGS_PATH.exists() else "mặc định")

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            LOGGER.info("Camera đã lên. Bấm 'q' hoặc Esc để tắt.")

            while True:
                success, image = cap.read()
                if not success:
                    LOGGER.error("Không đọc được frame từ camera.")
                    return 1

                image = cv2.flip(image, 1)
                draw_control_frame(image, config)

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                hand_result = landmarker.detect(mp_image)

                if hand_result.hand_landmarks:
                    for hand_landmarks in hand_result.hand_landmarks:
                        if not handle_hand(image, hand_landmarks, mouse_state, screen_size, config):
                            return 0
                else:
                    reset_gestures(mouse_state)
                    reset_pause_gesture(mouse_state)
                    if not release_drag(mouse_state):
                        return 0
                    if not mouse_state.control_enabled:
                        set_status(mouse_state, "Paused")

                if not ui_state.preview_hidden:
                    previous_time = draw_fps(image, previous_time)
                    draw_status(image, mouse_state, config, ui_state)
                    draw_help_panel(image, mouse_state)
                    draw_notice(image, ui_state)
                    cv2.imshow(WINDOW_NAME, image)
                    cv2_key = read_cv2_key()
                    if not handle_hotkey(cv2_key, config, mouse_state, ui_state):
                        break
                else:
                    if not process_keys(config, mouse_state, ui_state):
                        break
                    time.sleep(0.001)

                if not ui_state.preview_hidden and not process_keys(config, mouse_state, ui_state):
                    break

    finally:
        release_drag(mouse_state)
        cap.release()
        cv2.destroyAllWindows()

    return 0


def main() -> int:
    try:
        return run()
    except KeyboardInterrupt:
        LOGGER.info("Đã dừng bằng Ctrl+C.")
        return 0
    except Exception as error:
        LOGGER.exception("Lỗi khi chạy Hand Gesture Mouse: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
