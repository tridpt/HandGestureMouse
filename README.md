# Hand Gesture Mouse

Điều khiển chuột bằng camera và cử chỉ tay qua MediaPipe.

## Cách chạy

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py
```

## Cử chỉ

- Di chuyển ngón trỏ để điều khiển con trỏ chuột.
- Chạm đầu ngón trỏ và ngón giữa để click chuột trái.
- Chạm đầu ngón giữa và ngón áp út để click chuột phải.
- Chạm đầu ngón cái và ngón trỏ, giữ khoảng nửa giây để kéo thả.
- Giơ ngón trỏ + ngón giữa kiểu dấu V, gập các ngón còn lại, rồi đưa tay lên/xuống để scroll.
- Nắm tay và giữ một chút để bật/tắt điều khiển chuột.
- Bấm `q` hoặc `Esc` trong cửa sổ video để thoát.

## Phím tắt

- `+` / `-`: tăng/giảm độ mượt `smoothening`.
- `[` / `]`: giảm/tăng ngưỡng click trái.
- `,` / `.`: thu hẹp/mở rộng vùng điều khiển.
- `9` / `0`: giảm/tăng tốc độ scroll.
- `p`: bật/tắt điều khiển chuột thủ công.
- `m`: ẩn/hiện cửa sổ preview để chạy nền. Khi đã ẩn, bấm `m` trong terminal để hiện lại.
- `s`: lưu cấu hình hiện tại vào `settings.json`.
- `r`: đọc lại cấu hình từ `settings.json`.
- `q` hoặc `Esc`: thoát.

## Tinh chỉnh

Các thông số nằm trong `settings.json`, có thể sửa rồi chạy lại app.

- Chuột chậm quá: giảm `smoothening`, ví dụ `4.0`.
- Chuột rung quá: tăng `smoothening`, ví dụ `7.0`.
- Vùng điều khiển quá hẹp/rộng: chỉnh `control_frame_margin`.
- Click trái khó ăn: tăng `left_click_distance_ratio`.
- Click nhầm nhiều: giảm `left_click_distance_ratio`.
- Click bị lặp/nhấp nháy khi tay rung: tăng `left_click_release_ratio` / `right_click_release_ratio` (nới khoảng cách nhả) hoặc tăng `click_cooldown_seconds`.
- Click hai lần khi chỉ muốn một: tăng `click_cooldown_seconds`.
- Kéo thả khó kích hoạt: tăng `drag_distance_ratio` hoặc giảm `drag_hold_seconds`.
- Scroll quá nhanh/chậm: chỉnh `scroll_speed`.
- Pause bằng nắm tay quá nhạy/chậm: chỉnh `pause_fist_hold_seconds`.

## Kiểm tra logic

```powershell
python -m unittest
```

## Lưu ý an toàn

PyAutoGUI fail-safe đang được bật. Nếu cần dừng khẩn cấp, đưa chuột vật lý về góc màn hình hoặc bấm `Ctrl+C` trong terminal.

File `hand_landmarker.task` phải nằm cùng thư mục với `main.py`.
