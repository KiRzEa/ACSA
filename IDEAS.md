# LLM Preprocessing — Ideas (proposed, not yet implemented)

Nguồn động lực: teencode/lỗi chính tả, code-switching Việt-Anh, câu dài chứa
nhiều aspect, ý kiến ẩn/phủ định phức tạp, cách diễn đạt không đồng nhất giữa
domain — những vấn đề rule-based `clean_doc()` xử lý được một phần (đã fix
nhiều bug thực tế trong preprocessing.py), nhưng có giới hạn.

**Quyết định (2026-08-21)**: bỏ ý tưởng "LLM viết lại text 1 lượt, không kiểm
chứng" (Approach A cũ) — rủi ro làm sai nhãn gold mà không phát hiện được.
Lấy pipeline Generator/Verifier/Reflector làm **nền tảng chung**, áp dụng cho
mọi task LLM-preprocessing sau này, không riêng việc chuẩn hoá text.

## Pipeline gốc: Agentic Generator/Verifier/Reflector

**Giai đoạn 1 — Phát triển guideline (chạy trên 1 sample của TRAIN, có nhãn)**
1. Generator: sinh output theo guideline hiện tại (output là gì tuỳ task, xem 2 ví dụ bên dưới).
2. Verifier: đối chiếu output với nhãn gold — có mất/đổi/thêm thông tin sai lệch so với nhãn không?
3. Reflector: tổng hợp lỗi verifier tìm được theo batch, đề xuất cập nhật guideline để tránh lặp lại.
4. Lặp lại đến khi pass-rate ổn định/đạt ngưỡng (hội tụ).

**Giai đoạn 2 — Áp dụng đóng băng (frozen), không dùng nhãn**
Guideline cuối cùng (đã qua kiểm chứng ở Giai đoạn 1) áp dụng 1 lượt, không
lặp, không xem nhãn — lên toàn bộ Train, Dev, và **Test**. Test không bao giờ
lộ nhãn cho preprocessing, giữ đúng tính hợp lệ của đánh giá.

Gần với pattern "self-refine"/"actor-critic-reflection" trong literature
agentic LLM — nếu viết paper sau này có thể liên hệ tới nhóm work đó. Xem
diagram: https://claude.ai/code/artifact/631a624e-694f-49d7-9d3e-eaac1a57ad90

Cùng 1 khung Generator/Verifier/Reflector này, chỉ đổi **output type của
Generator** và cách Verifier đối chiếu, là dùng được cho các task khác nhau:

### Task 1 — Chuẩn hoá text

Generator viết lại câu cho sạch/rõ nghĩa hơn (teencode, lỗi chính tả,
code-switching), thay/bổ sung cho `clean_doc()`.

Ví dụ: `"Máy thì đẹp đấy mà pin như hạch, shop giao cũng lâu."` →
`"Điện thoại có thiết kế đẹp nhưng thời lượng pin kém và cửa hàng giao hàng chậm."`

Verifier: mỗi aspect trong nhãn còn được câu mới hỗ trợ rõ ràng, đúng polarity không?

### Task 2 — Structured extraction (làm input phụ, không sửa text gốc)

Generator trích xuất (entity, attribute, sentiment) trực tiếp từ câu:
```
- Điện thoại | thiết kế | đẹp | positive
- Điện thoại | pin | kém | negative
- Cửa hàng | giao hàng | chậm | negative
```
Verifier: so trực tiếp bộ triple trích xuất với nhãn gold (dễ đối chiếu hơn
Task 1 vì đã có cấu trúc sẵn — match/không match từng aspect). Encoder
(PhoBERT/XLM-R/Qwen) dự đoán dựa trên CẢ câu gốc lẫn structured output này —
câu gốc và nhãn gold không bị đụng tới, structured output chỉ là tín hiệu
input BỔ SUNG. Gần với đề xuất arch-v2 (entity/attribute auxiliary head) đã
bàn trước đó — khác là arch-v2 lấy entity/attribute từ nhãn gold có sẵn (gộp
coarse), còn cách này để LLM tự trích xuất trực tiếp từ câu.

**Đánh đổi cần quyết định**: nếu chỉ dùng structured output lúc TRAIN (input
phụ, cố định theo data đã có) → không cần LLM lúc inference, giống Task 1.
Nhưng nếu muốn dùng cho câu MỚI chưa từng thấy lúc inference → bắt buộc gọi
LLM mỗi lần predict, thêm cost/latency/dependency vào 1 API bên ngoài cho một
model vốn đang tự chứa (self-contained) trên PhoBERT.

## Việc cần làm trước khi implement

1. Quyết định làm Task 1, Task 2, hay cả 2 (có thể chạy song song, mỗi task 1 guideline riêng, cùng chung khung G/V/R).
2. Chọn model OpenAI nào (cân đối cost vs quality cho 7000+ mẫu/domain).
3. Chạy Giai đoạn 1 (phát triển guideline) trên sample nhỏ trước (~50-100 mẫu), không phải toàn bộ train — rẻ, nhanh, kiểm tra tay được.
4. Thiết kế script resumable (lưu tăng dần, tránh mất tiền/thời gian nếu bị ngắt giữa chừng).
