from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyautogui


MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
WINDOW_NAME = "Hand Gesture Mouse"
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CONTROL_FRAME_MARGIN = 100
SMOOTHENING = 7
CLICK_DISTANCE_RATIO = 0.12
SAFE_SCREEN_MARGIN = 2

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

LOGGER = logging.getLogger(__name__)


@dataclass
class MouseState:
    previous_x: float = 0.0
    previous_y: float = 0.0
    is_clicked: bool = False


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


def open_camera(index: int = CAMERA_INDEX) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Không mở được camera. Hãy kiểm tra quyền camera hoặc camera index.")

    return cap


def draw_landmarks(image, hand_landmarks) -> None:
    height, width, _ = image.shape
    for lm in hand_landmarks:
        cx, cy = landmark_to_pixel(lm, width, height)
        cv2.circle(image, (cx, cy), 4, (0, 255, 0), cv2.FILLED)


def landmark_to_pixel(landmark, width: int, height: int) -> tuple[int, int]:
    return int(landmark.x * width), int(landmark.y * height)


def hand_diagonal(hand_landmarks, width: int, height: int) -> float:
    xs = [lm.x * width for lm in hand_landmarks]
    ys = [lm.y * height for lm in hand_landmarks]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def map_camera_to_screen(
    x: int,
    y: int,
    image_width: int,
    image_height: int,
    screen_width: int,
    screen_height: int,
) -> tuple[float, float]:
    left = CONTROL_FRAME_MARGIN
    right = max(left + 1, image_width - CONTROL_FRAME_MARGIN)
    top = CONTROL_FRAME_MARGIN
    bottom = max(top + 1, image_height - CONTROL_FRAME_MARGIN)

    clamped_x = float(np.clip(x, left, right))
    clamped_y = float(np.clip(y, top, bottom))

    screen_x = np.interp(
        clamped_x,
        (left, right),
        (SAFE_SCREEN_MARGIN, max(SAFE_SCREEN_MARGIN, screen_width - SAFE_SCREEN_MARGIN)),
    )
    screen_y = np.interp(
        clamped_y,
        (top, bottom),
        (SAFE_SCREEN_MARGIN, max(SAFE_SCREEN_MARGIN, screen_height - SAFE_SCREEN_MARGIN)),
    )
    return float(screen_x), float(screen_y)


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


def click_mouse() -> bool:
    try:
        pyautogui.click()
    except pyautogui.FailSafeException:
        LOGGER.warning("Đã kích hoạt PyAutoGUI fail-safe. Dừng chương trình.")
        return False
    except Exception as error:
        LOGGER.warning("Không thể click chuột. Dừng chương trình: %s", error)
        return False
    return True


def handle_hand(image, hand_landmarks, state: MouseState, screen_size: tuple[int, int]) -> bool:
    if len(hand_landmarks) <= 12:
        state.is_clicked = False
        return True

    height, width, _ = image.shape
    screen_width, screen_height = screen_size

    draw_landmarks(image, hand_landmarks)

    index_tip = landmark_to_pixel(hand_landmarks[8], width, height)
    middle_tip = landmark_to_pixel(hand_landmarks[12], width, height)

    cv2.circle(image, index_tip, 15, (255, 0, 255), cv2.FILLED)
    cv2.circle(image, middle_tip, 15, (255, 0, 255), cv2.FILLED)

    target_x, target_y = map_camera_to_screen(
        index_tip[0],
        index_tip[1],
        width,
        height,
        screen_width,
        screen_height,
    )

    current_x = state.previous_x + (target_x - state.previous_x) / SMOOTHENING
    current_y = state.previous_y + (target_y - state.previous_y) / SMOOTHENING

    if not move_mouse(current_x, current_y):
        return False

    state.previous_x = current_x
    state.previous_y = current_y

    fingertip_distance = math.hypot(
        middle_tip[0] - index_tip[0],
        middle_tip[1] - index_tip[1],
    )
    pinch_ratio = fingertip_distance / max(hand_diagonal(hand_landmarks, width, height), 1.0)

    cv2.line(image, index_tip, middle_tip, (255, 0, 255), 3)
    if pinch_ratio < CLICK_DISTANCE_RATIO:
        cv2.circle(image, index_tip, 15, (0, 255, 0), cv2.FILLED)
        if not state.is_clicked:
            if not click_mouse():
                return False
            state.is_clicked = True
    else:
        state.is_clicked = False

    return True


def draw_control_frame(image) -> None:
    height, width, _ = image.shape
    cv2.rectangle(
        image,
        (CONTROL_FRAME_MARGIN, CONTROL_FRAME_MARGIN),
        (width - CONTROL_FRAME_MARGIN, height - CONTROL_FRAME_MARGIN),
        (255, 0, 255),
        2,
    )


def draw_fps(image, previous_time: float) -> float:
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


def run() -> int:
    configure_logging()

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0

    options = build_landmarker_options(MODEL_PATH)
    screen_size = tuple(pyautogui.size())
    state = MouseState()
    previous_time = time.time()
    cap = open_camera()

    LOGGER.info("Đang tải AI model nhận diện tay và mở camera...")

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            LOGGER.info("Camera đã lên. Bấm 'q' hoặc Esc ở cửa sổ video để tắt.")

            while True:
                success, image = cap.read()
                if not success:
                    LOGGER.error("Không đọc được frame từ camera.")
                    return 1

                image = cv2.flip(image, 1)
                draw_control_frame(image)

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                hand_result = landmarker.detect(mp_image)

                if hand_result.hand_landmarks:
                    for hand_landmarks in hand_result.hand_landmarks:
                        if not handle_hand(image, hand_landmarks, state, screen_size):
                            return 0
                else:
                    state.is_clicked = False

                previous_time = draw_fps(image, previous_time)
                cv2.imshow(WINDOW_NAME, image)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    finally:
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
