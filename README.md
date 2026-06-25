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
- Bấm `q` hoặc `Esc` trong cửa sổ video để thoát.

## Lưu ý an toàn

PyAutoGUI fail-safe đang được bật. Nếu cần dừng khẩn cấp, đưa chuột vật lý về góc màn hình hoặc bấm `Ctrl+C` trong terminal.

File `hand_landmarker.task` phải nằm cùng thư mục với `main.py`.
