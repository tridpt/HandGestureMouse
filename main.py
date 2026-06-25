from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import pyautogui


MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
SETTINGS_PATH = Path(__file__).with_name("settings.json")
WINDOW_NAME = "Hand Gesture Mouse"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppConfig:
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    control_frame_margin: int = 90
    smoothening: float = 5.0
    safe_screen_margin: int = 4
    left_click_distance_ratio: float = 0.12
    right_click_distance_ratio: float = 0.10
    drag_distance_ratio: float = 0.09
    drag_hold_seconds: float = 0.45
    scroll_spread_ratio: float = 0.18
    scroll_deadzone_px: int = 10
    scroll_speed: float = 0.35


@dataclass
class MouseState:
    previous_x: float = 0.0
    previous_y: float = 0.0
    left_click_active: bool = False
    right_click_active: bool = False
    drag_started_at: float | None = None
    is_dragging: bool = False
    scroll_anchor_y: float | None = None
    status: str = "Move"
    status_until: float = 0.0


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


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


def landmark_to_pixel(landmark: Any, width: int, height: int) -> tuple[int, int]:
    return int(landmark.x * width), int(landmark.y * height)


def hand_diagonal(hand_landmarks: list[Any], width: int, height: int) -> float:
    xs = [lm.x * width for lm in hand_landmarks]
    ys = [lm.y * height for lm in hand_landmarks]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def distance_ratio(
    point_a: tuple[int, int],
    point_b: tuple[int, int],
    diagonal: float,
) -> float:
    return math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1]) / max(diagonal, 1.0)


def is_finger_up(hand_landmarks: list[Any], tip_id: int, pip_id: int) -> bool:
    return hand_landmarks[tip_id].y < hand_landmarks[pip_id].y


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
    if state.status_until and time.monotonic() > state.status_until:
        return "Move"
    return state.status


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


def draw_status(image: Any, state: MouseState) -> None:
    status = current_status(state)
    cv2.rectangle(image, (14, 68), (260, 112), (20, 20, 20), cv2.FILLED)
    cv2.putText(
        image,
        f"Mode: {status}",
        (24, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
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
    now: float,
) -> bool:
    if thumb_index_ratio >= config.drag_distance_ratio:
        return release_drag(state)

    state.left_click_active = False
    state.right_click_active = False
    state.scroll_anchor_y = None

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


def is_scroll_gesture(
    hand_landmarks: list[Any],
    index_middle_ratio: float,
    config: AppConfig,
) -> bool:
    index_up = is_finger_up(hand_landmarks, 8, 6)
    middle_up = is_finger_up(hand_landmarks, 12, 10)
    ring_down = not is_finger_up(hand_landmarks, 16, 14)
    pinky_down = not is_finger_up(hand_landmarks, 20, 18)
    return index_up and middle_up and ring_down and pinky_down and (
        index_middle_ratio > config.scroll_spread_ratio
    )


def handle_hand(
    image: Any,
    hand_landmarks: list[Any],
    state: MouseState,
    screen_size: tuple[int, int],
    config: AppConfig,
) -> bool:
    if len(hand_landmarks) <= 20:
        reset_gestures(state)
        return release_drag(state)

    height, width, _ = image.shape
    screen_width, screen_height = screen_size

    draw_landmarks(image, hand_landmarks)

    thumb_tip = landmark_to_pixel(hand_landmarks[4], width, height)
    index_tip = landmark_to_pixel(hand_landmarks[8], width, height)
    middle_tip = landmark_to_pixel(hand_landmarks[12], width, height)
    ring_tip = landmark_to_pixel(hand_landmarks[16], width, height)

    cv2.circle(image, index_tip, 15, (255, 0, 255), cv2.FILLED)
    cv2.circle(image, middle_tip, 12, (255, 0, 255), cv2.FILLED)
    cv2.circle(image, ring_tip, 10, (255, 0, 255), cv2.FILLED)

    target_x, target_y = map_camera_to_screen(
        index_tip[0],
        index_tip[1],
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

    diagonal = hand_diagonal(hand_landmarks, width, height)
    thumb_index_ratio = distance_ratio(thumb_tip, index_tip, diagonal)
    index_middle_ratio = distance_ratio(index_tip, middle_tip, diagonal)
    middle_ring_ratio = distance_ratio(middle_tip, ring_tip, diagonal)

    now = time.monotonic()
    if thumb_index_ratio < config.drag_distance_ratio or state.drag_started_at is not None:
        cv2.line(image, thumb_tip, index_tip, (0, 165, 255), 3)
        return handle_drag(state, config, thumb_index_ratio, now)

    if not handle_clicks(state, index_middle_ratio, middle_ring_ratio, config):
        return False

    if state.left_click_active or state.right_click_active:
        return True

    if is_scroll_gesture(hand_landmarks, index_middle_ratio, config):
        return handle_scroll(image, state, config, index_tip, middle_tip)

    state.scroll_anchor_y = None
    if not state.status_until:
        set_status(state, "Move")
    return True


def run() -> int:
    configure_logging()

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0

    config = load_config()
    options = build_landmarker_options(MODEL_PATH)
    screen_size = tuple(pyautogui.size())
    state = MouseState()
    previous_time = time.time()
    cap = open_camera(config)

    LOGGER.info("Đang tải AI model nhận diện tay và mở camera...")
    LOGGER.info("Đang dùng cấu hình: %s", SETTINGS_PATH.name if SETTINGS_PATH.exists() else "mặc định")

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            LOGGER.info("Camera đã lên. Bấm 'q' hoặc Esc ở cửa sổ video để tắt.")

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
                        if not handle_hand(image, hand_landmarks, state, screen_size, config):
                            return 0
                else:
                    reset_gestures(state)
                    if not release_drag(state):
                        return 0

                previous_time = draw_fps(image, previous_time)
                draw_status(image, state)
                cv2.imshow(WINDOW_NAME, image)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    finally:
        release_drag(state)
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
