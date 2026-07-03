from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

from app_config import AppConfig
from gesture_logic import analyze_hand, distance_ratio, is_fist_gesture, is_scroll_gesture


@dataclass
class Landmark:
    x: float
    y: float


def rotate_hand(hand: list[Landmark], degrees: float, pivot: Landmark) -> list[Landmark]:
    theta = math.radians(degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rotated = []
    for point in hand:
        dx, dy = point.x - pivot.x, point.y - pivot.y
        rotated.append(
            Landmark(
                pivot.x + dx * cos_t - dy * sin_t,
                pivot.y + dx * sin_t + dy * cos_t,
            )
        )
    return rotated


def make_hand(*, index_up: bool, middle_up: bool, ring_up: bool, pinky_up: bool) -> list[Landmark]:
    points = [Landmark(0.5, 0.8) for _ in range(21)]

    points[4] = Landmark(0.34, 0.58)
    points[6] = Landmark(0.42, 0.55)
    points[8] = Landmark(0.38, 0.30 if index_up else 0.72)
    points[10] = Landmark(0.50, 0.55)
    points[12] = Landmark(0.54, 0.30 if middle_up else 0.72)
    points[14] = Landmark(0.62, 0.55)
    points[16] = Landmark(0.66, 0.30 if ring_up else 0.72)
    points[18] = Landmark(0.74, 0.55)
    points[20] = Landmark(0.78, 0.30 if pinky_up else 0.72)
    return points


class GestureLogicTest(unittest.TestCase):
    def test_distance_ratio_uses_hand_diagonal(self) -> None:
        self.assertAlmostEqual(distance_ratio((0, 0), (3, 4), 10), 0.5)

    def test_fist_when_all_four_fingers_are_down(self) -> None:
        hand = make_hand(index_up=False, middle_up=False, ring_up=False, pinky_up=False)
        metrics = analyze_hand(hand, 640, 480)
        self.assertTrue(is_fist_gesture(metrics))

    def test_not_fist_when_index_is_up(self) -> None:
        hand = make_hand(index_up=True, middle_up=False, ring_up=False, pinky_up=False)
        metrics = analyze_hand(hand, 640, 480)
        self.assertFalse(is_fist_gesture(metrics))

    def test_scroll_gesture_requires_v_shape_with_ring_and_pinky_down(self) -> None:
        hand = make_hand(index_up=True, middle_up=True, ring_up=False, pinky_up=False)
        metrics = analyze_hand(hand, 640, 480)
        config = AppConfig(scroll_spread_ratio=0.12)
        self.assertTrue(is_scroll_gesture(metrics, config))

    def test_scroll_gesture_rejects_open_hand(self) -> None:
        hand = make_hand(index_up=True, middle_up=True, ring_up=True, pinky_up=True)
        metrics = analyze_hand(hand, 640, 480)
        self.assertFalse(is_scroll_gesture(metrics, AppConfig()))

    def test_scroll_gesture_survives_hand_rotation(self) -> None:
        wrist = Landmark(0.5, 0.8)
        base = make_hand(index_up=True, middle_up=True, ring_up=False, pinky_up=False)
        config = AppConfig(scroll_spread_ratio=0.12)
        for degrees in (-60, -30, 30, 60, 90):
            rotated = rotate_hand(base, degrees, wrist)
            metrics = analyze_hand(rotated, 640, 480)
            self.assertTrue(
                is_scroll_gesture(metrics, config),
                f"Cử chỉ scroll phải nhận diện được khi tay xoay {degrees} độ",
            )

    def test_fist_survives_hand_rotation(self) -> None:
        wrist = Landmark(0.5, 0.8)
        base = make_hand(index_up=False, middle_up=False, ring_up=False, pinky_up=False)
        for degrees in (-60, -30, 30, 60, 90):
            rotated = rotate_hand(base, degrees, wrist)
            metrics = analyze_hand(rotated, 640, 480)
            self.assertTrue(
                is_fist_gesture(metrics),
                f"Nắm tay phải nhận diện được khi tay xoay {degrees} độ",
            )


if __name__ == "__main__":
    unittest.main()
