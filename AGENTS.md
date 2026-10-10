# CLAUDE.md

## Tổng quan
Collaborative-Filtering-System là hệ thống **gợi ý sách bằng lọc cộng tác** (môn Nhập môn Trí tuệ nhân tạo — BTL), dùng bộ dữ liệu Book-Crossing (Books, Users, Ratings). Có hai hướng: (1) lọc cộng tác dựa trên item bằng **cosine similarity** trên ma trận sách × người dùng; (2) mô hình **SVD** (thư viện `surprise`) dự đoán điểm đánh giá.

## Cấu trúc
- `Model.py` — pipeline cosine-similarity: ghép dữ liệu, chỉ giữ người dùng có >200 đánh giá và sách có >=50 đánh giá, dựng pivot table, tự cài `cosine_similarity`; hàm `recommend(book_name)` trả các sách tương tự.
- `Model2.py` — biến thể thứ hai của hệ gợi ý (có hàm `recommend(book_name)`), cài thêm các phép tính tự viết.
- `web.py` — ứng dụng Flask: form tìm tên sách → gọi `Model.recommend` → hiển thị kết quả. **Cần `templates/index.html` nhưng file này không có trong repo.**
- `CodeKaggle.ipynb` — notebook khám phá dữ liệu + lọc cộng tác (dựa trên notebook Kaggle).
- `codetay.ipynb` — thử nghiệm SVD (`surprise`), RMSE, top-10 gợi ý cho một người dùng mẫu.
- `Books.csv.zip`, `Ratings.csv.zip`, `Users.csv.zip` — dữ liệu gốc (nén, ~25MB). Cần giải nén ra `Books.csv`, `Ratings.csv`, `Users.csv` trước khi chạy.

## Môi trường
Python 3.12: `pandas numpy scikit-learn scipy matplotlib seaborn flask scikit-surprise`. Chạy web: `python web.py` (debug mode, cổng 5000).

## Lưu ý quan trọng
- Các script/notebook đọc dữ liệu bằng **đường dẫn tuyệt đối của máy tác giả** (`S:\File\School\NMTTNT\BTL\Data\*.csv`) — phải đổi sang đường dẫn tương đối khi chạy trên máy khác.
- `Model.py` tính toàn bộ ma trận ngay khi import (chậm, tốn RAM); `web.py` bắt `except:` trống và nuốt lỗi.
- `web.py` chạy `debug=True` — không dùng để deploy thật.

## Quy tắc khi làm việc
- Đây là bài làm môn học: không tự ý refactor hoặc đổi kết quả mô hình khi chưa được yêu cầu.
- Không commit dữ liệu giải nén (`*.csv` đã nằm trong `.gitignore`), khóa API hay thông tin cá nhân.
- Dữ liệu người dùng Book-Crossing đã ẩn danh nhưng vẫn có cột `Location`/`Age`; không ghép thêm nguồn nhằm định danh người dùng.
