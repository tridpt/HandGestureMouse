from __future__ import annotations

import unittest
from unittest import mock

from app_config import AppConfig

# main.py imports cv2/mediapipe/pyautogui at module load. Skip cleanly if the
# runtime deps are not installed in the current environment.
try:
    import main
except Exception as import_error:  # pragma: no cover - environment dependent.
    main = None
    _IMPORT_ERROR = import_error
else:
    _IMPORT_ERROR = None


@unittest.skipIf(main is None, f"main.py không import được: {_IMPORT_ERROR}")
class HandleClicksTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig(
            left_click_distance_ratio=0.12,
            left_click_release_ratio=0.16,
            right_click_distance_ratio=0.10,
            right_click_release_ratio=0.14,
            click_cooldown_seconds=0.35,
        )
        self.state = main.MouseState()
        # Freeze monotonic clock so cooldown behaviour is deterministic.
        self.now = 100.0
        patcher = mock.patch.object(main.time, "monotonic", side_effect=lambda: self.now)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _clicks_with(self, index_middle: float, middle_ring: float = 0.5) -> mock.Mock:
        with mock.patch.object(main, "click_mouse", return_value=True) as click:
            main.handle_clicks(self.state, index_middle, middle_ring, self.config)
        return click

    def test_left_click_fires_once_on_pinch(self) -> None:
        click = self._clicks_with(index_middle=0.08)
        click.assert_called_once_with("left")
        self.assertTrue(self.state.left_click_active)

    def test_jitter_at_threshold_does_not_refire(self) -> None:
        # First pinch fires the click.
        self._clicks_with(index_middle=0.08)
        # Value bounces just above the press threshold but below release: no new click.
        click = self._clicks_with(index_middle=0.13)
        click.assert_not_called()
        self.assertTrue(self.state.left_click_active)

    def test_release_requires_crossing_release_threshold(self) -> None:
        self._clicks_with(index_middle=0.08)
        # Spread past the release threshold clears the active flag.
        self._clicks_with(index_middle=0.17)
        self.assertFalse(self.state.left_click_active)

    def test_cooldown_blocks_rapid_second_click(self) -> None:
        self._clicks_with(index_middle=0.08)
        # Release, then pinch again immediately (within cooldown window).
        self._clicks_with(index_middle=0.17)
        self.now += 0.10  # less than click_cooldown_seconds
        click = self._clicks_with(index_middle=0.08)
        click.assert_not_called()

    def test_second_click_allowed_after_cooldown(self) -> None:
        self._clicks_with(index_middle=0.08)
        self._clicks_with(index_middle=0.17)
        self.now += 0.40  # past cooldown
        click = self._clicks_with(index_middle=0.08)
        click.assert_called_once_with("left")

    def test_right_click_fires_on_middle_ring_pinch(self) -> None:
        with mock.patch.object(main, "click_mouse", return_value=True) as click:
            main.handle_clicks(self.state, 0.5, 0.06, self.config)
        click.assert_called_once_with("right")
        self.assertTrue(self.state.right_click_active)


if __name__ == "__main__":
    unittest.main()
