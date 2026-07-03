from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from app_config import AppConfig


@dataclass(frozen=True)
class GestureMetrics:
    thumb_tip: tuple[int, int]
    index_tip: tuple[int, int]
    middle_tip: tuple[int, int]
    ring_tip: tuple[int, int]
    diagonal: float
    thumb_index_ratio: float
    index_middle_ratio: float
    middle_ring_ratio: float
    index_up: bool
    middle_up: bool
    ring_up: bool
    pinky_up: bool


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


WRIST_ID = 0


def _distance(point_a: Any, point_b: Any) -> float:
    return math.hypot(point_a.x - point_b.x, point_a.y - point_b.y)


def is_finger_up(hand_landmarks: list[Any], tip_id: int, pip_id: int) -> bool:
    """Ngón được coi là duỗi khi đầu ngón xa cổ tay hơn khớp PIP.

    So sánh khoảng cách tới cổ tay (landmark 0) thay vì chỉ so tọa độ y, nên
    kết quả không phụ thuộc vào việc tay đang thẳng đứng, nghiêng hay xoay.
    """
    wrist = hand_landmarks[WRIST_ID]
    tip_distance = _distance(hand_landmarks[tip_id], wrist)
    pip_distance = _distance(hand_landmarks[pip_id], wrist)
    return tip_distance > pip_distance


def analyze_hand(hand_landmarks: list[Any], width: int, height: int) -> GestureMetrics:
    if len(hand_landmarks) <= 20:
        raise ValueError("Cần đủ 21 hand landmarks để phân tích cử chỉ.")

    diagonal = hand_diagonal(hand_landmarks, width, height)
    thumb_tip = landmark_to_pixel(hand_landmarks[4], width, height)
    index_tip = landmark_to_pixel(hand_landmarks[8], width, height)
    middle_tip = landmark_to_pixel(hand_landmarks[12], width, height)
    ring_tip = landmark_to_pixel(hand_landmarks[16], width, height)

    return GestureMetrics(
        thumb_tip=thumb_tip,
        index_tip=index_tip,
        middle_tip=middle_tip,
        ring_tip=ring_tip,
        diagonal=diagonal,
        thumb_index_ratio=distance_ratio(thumb_tip, index_tip, diagonal),
        index_middle_ratio=distance_ratio(index_tip, middle_tip, diagonal),
        middle_ring_ratio=distance_ratio(middle_tip, ring_tip, diagonal),
        index_up=is_finger_up(hand_landmarks, 8, 6),
        middle_up=is_finger_up(hand_landmarks, 12, 10),
        ring_up=is_finger_up(hand_landmarks, 16, 14),
        pinky_up=is_finger_up(hand_landmarks, 20, 18),
    )


def is_fist_gesture(metrics: GestureMetrics) -> bool:
    return not (
        metrics.index_up
        or metrics.middle_up
        or metrics.ring_up
        or metrics.pinky_up
    )


def is_scroll_gesture(metrics: GestureMetrics, config: AppConfig) -> bool:
    return (
        metrics.index_up
        and metrics.middle_up
        and not metrics.ring_up
        and not metrics.pinky_up
        and metrics.index_middle_ratio > config.scroll_spread_ratio
    )
