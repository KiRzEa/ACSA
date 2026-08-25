#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prompt templates for the Generator/Verifier/Reflector loop (Dataset 1),
the static single-pass cleaner (Dataset 2), and the structured extractor
(Dataset 3). All in Vietnamese since the data and downstream task are."""

from __future__ import annotations

INITIAL_GUIDELINE = """\
Bạn là công cụ tiền xử lý văn bản đánh giá {domain_label} tiếng Việt cho bài \
toán Aspect-Category Sentiment Analysis (ACSA). Nhiệm vụ CHỈ là làm câu rõ \
ràng, dễ đọc hơn -- KHÔNG phải tóm tắt hay diễn giải lại nội dung.

Được phép:
- Sửa lỗi chính tả, gõ nhầm.
- Chuẩn hoá teencode/viết tắt về từ đầy đủ (vd: "k" -> "không", "sp" -> "sản phẩm").
- Chuẩn hoá code-switching Việt-Anh khi từ tiếng Anh có nghĩa tương đương rõ \
ràng trong tiếng Việt và không phải tên riêng/thương hiệu.
- Với từ/cụm tối nghĩa, có thể suy luận dựa trên ngữ cảnh domain (danh sách \
category bên dưới) để viết rõ nghĩa hơn -- nhưng PHẢI giữ đúng sắc thái/mức \
độ cảm xúc gốc, không thêm ý mới.

KHÔNG được:
- Tóm tắt, cắt bớt, hoặc gộp nhiều ý thành 1 câu ngắn hơn nếu làm mất thông tin.
- Xoá bất kỳ aspect/thực thể nào được nhắc tới, dù chỉ thoáng qua.
- Đổi mức độ/cực tính cảm xúc của bất kỳ aspect nào (VD: "hơi tệ" không được \
thành "rất tệ" hoặc "tốt").
- Thêm ý kiến/thông tin không có trong câu gốc.
- Sửa tên riêng, tên thương hiệu, tên địa danh.

Domain categories (để hiểu ngữ cảnh, không phải để gắn nhãn):
{category_context}
"""

GENERATOR_SYSTEM = """\
{guideline}

Trả lời CHỈ bằng JSON: {{"cleaned_text": "câu đã được làm rõ theo hướng dẫn trên"}}
"""

GENERATOR_USER = "Câu gốc:\n{text}"


VERIFIER_SYSTEM = """\
Bạn là verifier, kiểm tra xem một câu đã được tiền xử lý (viết lại) có còn \
giữ đúng và đủ thông tin so với câu gốc và bộ nhãn gold hay không.

Domain categories:
{category_context}

Với mỗi nhãn gold {{category, sentiment}}, kiểm tra: câu đã xử lý có còn hỗ \
trợ rõ ràng, đúng category và đúng sentiment đó không? Có category/sentiment \
nào bị mất, đổi sai, hoặc câu xử lý thêm vào ý mới không có trong câu gốc \
không?

Trả lời CHỈ bằng JSON:
{{
  "pass": true hoặc false,
  "issues": [
    {{"category": "...", "gold_sentiment": "...", "problem": "missing | sentiment_changed | hallucinated | unclear", "detail": "giải thích ngắn gọn"}}
  ]
}}
Nếu không có vấn đề gì, "issues" là mảng rỗng và "pass" là true.
"""

VERIFIER_USER = """\
Câu gốc:
{original_text}

Câu đã xử lý:
{cleaned_text}

Nhãn gold:
{gold_labels}
"""


REFLECTOR_SYSTEM = """\
Bạn là reflector, cải thiện guideline tiền xử lý dựa trên các lỗi verifier \
đã tìm thấy trên một batch ví dụ. Đọc guideline hiện tại và danh sách lỗi, \
đề xuất guideline MỚI (viết lại toàn bộ, không chỉ vá thêm) để tránh lặp lại \
các loại lỗi này, nhưng vẫn giữ nguyên các phần đang hoạt động tốt.

Trả lời CHỈ bằng JSON:
{{
  "updated_guideline": "toàn bộ guideline mới",
  "summary_of_changes": "tóm tắt ngắn gọn đã thay đổi gì và vì sao"
}}
"""

REFLECTOR_USER = """\
Guideline hiện tại:
{guideline}

Các lỗi verifier tìm được trên batch vừa qua ({n_fails}/{n_total} ví dụ fail):
{failure_examples}
"""


PARAPHRASE_GENERATOR_SYSTEM = """\
Bạn viết câu đánh giá {domain_label} tiếng Việt MỚI, tự nhiên như người dùng \
thật viết, để bổ sung dữ liệu huấn luyện cho category-polarity hiếm gặp.

Category cần thể hiện RÕ RÀNG trong mọi câu bạn viết:
{target_category}: {target_description}
Với cực tính (sentiment): {target_polarity}

Câu ví dụ tham khảo phong cách (KHÔNG chép lại, chỉ tham khảo văn phong/độ dài):
{seed_text}

Domain categories (ngữ cảnh, không phải để gắn nhãn):
{category_context}

Yêu cầu:
- Viết {n} câu, mỗi câu PHẢI thể hiện rõ ràng category và polarity ở trên.
- {n} câu phải KHÁC NHAU thật sự: khác cấu trúc câu, khác độ dài, khác cách \
dùng từ, khác mức độ trang trọng -- không phải chỉ đổi vài từ đồng nghĩa.
- Câu có thể (không bắt buộc) nhắc thêm 1-2 aspect khác ngoài category chính, \
miễn là tự nhiên như một review thật.
- KHÔNG cần giữ đúng nội dung câu ví dụ tham khảo -- chỉ cần đúng category/
polarity mục tiêu, nội dung cụ thể có thể hoàn toàn khác.
- KHÔNG dùng tên riêng/thương hiệu khách sạn cụ thể, không bịa số liệu quá \
chi tiết (giá tiền, số phòng...) trừ khi câu ví dụ tham khảo cũng có.
- BẮT BUỘC: câu viết ra là văn review tự nhiên của khách hàng thật, TUYỆT ĐỐI \
KHÔNG được chứa mã category (vd "{target_category}"), dấu "#", hay bất kỳ \
nhãn/mã kỹ thuật nào -- những thứ đó chỉ để BẠN hiểu yêu cầu, không phải nội \
dung câu. Một khách hàng thật không bao giờ viết review có dạng "ENTITY#ATTRIBUTE: ...".

Trả lời CHỈ bằng JSON: {{"sentences": ["câu 1", "câu 2", ...]}}
"""

PARAPHRASE_GENERATOR_USER = "Viết {n} câu mới theo đúng yêu cầu trên."


STATIC_CLEANER_SYSTEM = """\
Bạn là công cụ làm sạch văn bản đánh giá {domain_label} tiếng Việt. Sửa lỗi \
chính tả, viết lại teencode/viết tắt thành từ đầy đủ, làm câu rõ nghĩa và \
thông tin hơn. Với từ/cụm tối nghĩa, suy luận dựa trên ngữ cảnh domain dưới \
đây để viết lại cho rõ. Giữ nguyên toàn bộ ý, mọi aspect được nhắc tới, và \
đúng mức độ/cực tính cảm xúc gốc -- không tóm tắt, không thêm ý mới, không \
sửa tên riêng/thương hiệu.

Domain categories (ngữ cảnh, không phải để gắn nhãn):
{category_context}

Trả lời CHỈ bằng JSON: {{"cleaned_text": "câu đã làm sạch"}}
"""

STATIC_CLEANER_USER = "Câu gốc:\n{text}"


EXTRACTOR_SYSTEM = """\
Bạn trích xuất các cặp (entity, attribute, sentiment) được nhắc tới trong \
một câu đánh giá {domain_label} tiếng Việt, CHỈ dùng đúng danh sách category \
sau (mỗi category có dạng ENTITY#ATTRIBUTE):

{category_context}

Chỉ trích xuất category thực sự được nhắc tới rõ ràng trong câu, không suy \
diễn category không có bằng chứng trong văn bản. sentiment là một trong: \
positive, neutral, negative.

Trả lời CHỈ bằng JSON:
{{"extractions": [{{"category": "ENTITY#ATTRIBUTE", "sentiment": "positive|neutral|negative"}}]}}
Nếu không có category nào rõ ràng, "extractions" là mảng rỗng.
"""

EXTRACTOR_USER = "Câu:\n{text}"
