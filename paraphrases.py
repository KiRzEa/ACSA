"""Paraphrases of mapper.CATEGORY_DESCRIPTIONS, for the query-paraphrase-robustness experiment
(scripts/paraphrase_robustness.py). For every category in every domain CAGE trains on, 2 semantically
equivalent rewordings of the original description -- same aspect, same scope, different surface form
(synonym substitution, clause reordering, and a shift from noun-phrase listing to an evaluative-question
register for variant 2) -- so scripts/paraphrase_robustness.py can test whether CAGE's per-category
prediction is stable when the query text is reworded without changing what it refers to.

Structure mirrors mapper.CATEGORY_DESCRIPTIONS exactly: CATEGORY_PARAPHRASES[domain][category_code] =
[paraphrase_1, paraphrase_2]. Keys are the same domain names (Restaurant/Hotel/Phone/Education/Beauty)
and the same category codes (mapper._DESCRIBED_DOMAINS), so get_paraphrases() below can reuse
mapper._resolve_key() unchanged to bridge Education/Beauty's Vietnamese data-file category names.
"""
from mapper import CATEGORY_DESCRIPTIONS, _resolve_key, _DESCRIBED_DOMAINS  # noqa: F401

CATEGORY_PARAPHRASES = {
    "Restaurant": {
        "AMBIENCE#GENERAL": [
            "bầu không khí, cách bài trí và mức độ thoải mái chung của quán",
            "Quán có không gian dễ chịu, trang trí đẹp và thoải mái hay không?",
        ],
        "DRINKS#PRICES": [
            "mức giá và độ đắt rẻ của thức uống",
            "Đồ uống ở đây có giá cả hợp lý hay đắt đỏ so với mặt bằng chung?",
        ],
        "DRINKS#QUALITY": [
            "độ ngon, hương vị và chất lượng chung của thức uống",
            "Thức uống pha có ngon, đậm vị và chất lượng hay không?",
        ],
        "DRINKS#STYLE&OPTIONS": [
            "sự đa dạng lựa chọn, cách pha chế, kích cỡ và topping của thức uống",
            "Menu đồ uống có nhiều lựa chọn, kích cỡ và topping phong phú hay không?",
        ],
        "FOOD#PRICES": [
            "mức giá và độ đắt rẻ của món ăn",
            "Món ăn ở đây có giá cả hợp lý hay đắt đỏ so với mặt bằng chung?",
        ],
        "FOOD#QUALITY": [
            "độ tươi ngon, hương vị và cảm nhận chung về món ăn",
            "Món ăn có ngon, tươi và để lại cảm nhận tốt hay không?",
        ],
        "FOOD#STYLE&OPTIONS": [
            "sự đa dạng lựa chọn món, cách chế biến và khẩu phần",
            "Thực đơn có nhiều món để chọn, chế biến đa dạng và khẩu phần hợp lý hay không?",
        ],
        "LOCATION#GENERAL": [
            "địa điểm và mức độ dễ tìm, thuận tiện di chuyển của quán",
            "Quán nằm ở vị trí thuận tiện, dễ tìm hay không?",
        ],
        "RESTAURANT#GENERAL": [
            "trải nghiệm tổng thể và đánh giá chung dành cho nhà hàng hoặc quán",
            "Nhìn chung, trải nghiệm ở nhà hàng hoặc quán này có tốt hay không?",
        ],
        "RESTAURANT#MISCELLANEOUS": [
            "các yếu tố khác của quán như tiện ích, giao hàng, giữ xe, vệ sinh, khuyến mãi",
            "Quán còn có điểm gì khác đáng chú ý, như giao hàng, giữ xe, vệ sinh hay khuyến mãi?",
        ],
        "RESTAURANT#PRICES": [
            "độ đáng tiền, hóa đơn và mức giá chung của nhà hàng hoặc quán",
            "Chi tiêu ở nhà hàng hoặc quán này có xứng đáng với số tiền bỏ ra hay không?",
        ],
        "SERVICE#GENERAL": [
            "sự chuyên nghiệp, tốc độ và thái độ phục vụ của nhân viên",
            "Nhân viên phục vụ có nhanh nhẹn, chuyên nghiệp và thái độ tốt hay không?",
        ],
    },
    "Hotel": {
        "FACILITIES#CLEANLINESS": [
            "độ sạch sẽ và vệ sinh của các tiện ích, cơ sở vật chất chung",
            "Các khu vực tiện ích chung của khách sạn có được giữ sạch sẽ hay không?",
        ],
        "FACILITIES#COMFORT": [
            "cảm giác dễ chịu khi sử dụng các cơ sở vật chất và tiện ích chung",
            "Sử dụng các tiện ích chung của khách sạn có thoải mái hay không?",
        ],
        "FACILITIES#DESIGN&FEATURES": [
            "kiến trúc, thiết kế và các tính năng của tiện ích chung",
            "Các cơ sở vật chất chung được thiết kế và trang bị tính năng ra sao?",
        ],
        "FACILITIES#GENERAL": [
            "nhận xét chung về cơ sở vật chất và tiện ích của khách sạn",
            "Nhìn chung, cơ sở vật chất và tiện ích của khách sạn có tốt hay không?",
        ],
        "FACILITIES#MISCELLANEOUS": [
            "các vấn đề khác của cơ sở vật chất, chẳng hạn wifi, thang máy, bãi đỗ xe",
            "Cơ sở vật chất còn điểm gì khác đáng chú ý, như wifi, thang máy hay chỗ đậu xe?",
        ],
        "FACILITIES#PRICES": [
            "mức phí và chi phí sử dụng các cơ sở vật chất, tiện ích",
            "Sử dụng các tiện ích, cơ sở vật chất có tốn phí cao hay không?",
        ],
        "FACILITIES#QUALITY": [
            "tình trạng hoạt động và chất lượng của cơ sở vật chất, tiện ích",
            "Các cơ sở vật chất, tiện ích chung có hoạt động tốt và chất lượng hay không?",
        ],
        "FOOD&DRINKS#MISCELLANEOUS": [
            "các vấn đề khác về đồ ăn thức uống, như giờ phục vụ, cách sắp xếp buffet",
            "Đồ ăn thức uống còn điểm gì khác đáng chú ý, như giờ phục vụ hay cách bày biện?",
        ],
        "FOOD&DRINKS#PRICES": [
            "mức giá và độ đắt rẻ của đồ ăn thức uống tại khách sạn",
            "Đồ ăn thức uống tại khách sạn có giá cả hợp lý hay đắt đỏ?",
        ],
        "FOOD&DRINKS#QUALITY": [
            "độ ngon, hương vị và chất lượng chung của đồ ăn thức uống",
            "Đồ ăn thức uống tại khách sạn có ngon và chất lượng hay không?",
        ],
        "FOOD&DRINKS#STYLE&OPTIONS": [
            "loại hình, cách chế biến và sự đa dạng của đồ ăn thức uống",
            "Đồ ăn thức uống tại khách sạn có nhiều lựa chọn và chế biến đa dạng hay không?",
        ],
        "HOTEL#CLEANLINESS": [
            "mức độ vệ sinh, sạch sẽ chung của toàn bộ khách sạn",
            "Khách sạn nhìn chung có sạch sẽ, vệ sinh hay không?",
        ],
        "HOTEL#COMFORT": [
            "cảm giác dễ chịu, thoải mái khi lưu trú tại khách sạn",
            "Lưu trú tại khách sạn này có mang lại cảm giác thoải mái hay không?",
        ],
        "HOTEL#DESIGN&FEATURES": [
            "các tính năng nổi bật, kiến trúc và thiết kế của khách sạn",
            "Khách sạn được thiết kế và có những tính năng nổi bật nào?",
        ],
        "HOTEL#GENERAL": [
            "trải nghiệm tổng thể và đánh giá chung về khách sạn",
            "Nhìn chung, trải nghiệm tại khách sạn này có tốt hay không?",
        ],
        "HOTEL#MISCELLANEOUS": [
            "các vấn đề khác của khách sạn không thuộc khía cạnh cụ thể nào",
            "Khách sạn còn điểm gì khác đáng chú ý ngoài những khía cạnh đã nêu?",
        ],
        "HOTEL#PRICES": [
            "độ đáng tiền, hóa đơn và mức giá chung của khách sạn",
            "Chi phí lưu trú tại khách sạn này có xứng đáng với số tiền bỏ ra hay không?",
        ],
        "HOTEL#QUALITY": [
            "trải nghiệm lưu trú và chất lượng dịch vụ tại khách sạn",
            "Chất lượng dịch vụ và trải nghiệm lưu trú tại khách sạn này ra sao?",
        ],
        "LOCATION#GENERAL": [
            "vị trí và mức độ thuận tiện di chuyển của khách sạn",
            "Khách sạn có nằm ở vị trí thuận tiện, dễ di chuyển hay không?",
        ],
        "ROOMS#CLEANLINESS": [
            "mức độ vệ sinh, sạch sẽ của phòng nghỉ",
            "Phòng nghỉ có được dọn dẹp sạch sẽ hay không?",
        ],
        "ROOMS#COMFORT": [
            "cảm giác dễ chịu, thoải mái khi nghỉ ngơi trong phòng",
            "Nghỉ ngơi trong phòng có mang lại cảm giác thoải mái hay không?",
        ],
        "ROOMS#DESIGN&FEATURES": [
            "cách bài trí, thiết kế và các tính năng của phòng nghỉ",
            "Phòng nghỉ được thiết kế và bài trí như thế nào?",
        ],
        "ROOMS#GENERAL": [
            "nhận xét chung về phòng nghỉ",
            "Nhìn chung, phòng nghỉ có tốt hay không?",
        ],
        "ROOMS#MISCELLANEOUS": [
            "các vấn đề khác của phòng nghỉ không thuộc khía cạnh cụ thể nào",
            "Phòng nghỉ còn điểm gì khác đáng chú ý ngoài những khía cạnh đã nêu?",
        ],
        "ROOMS#PRICES": [
            "mức phí và chi phí của phòng nghỉ",
            "Giá phòng nghỉ có hợp lý hay đắt đỏ?",
        ],
        "ROOMS#QUALITY": [
            "tình trạng và chất lượng chung của phòng nghỉ",
            "Phòng nghỉ có chất lượng tốt và tình trạng ổn hay không?",
        ],
        "ROOM_AMENITIES#CLEANLINESS": [
            "mức độ vệ sinh, sạch sẽ của các tiện nghi trong phòng",
            "Các tiện nghi trong phòng có được giữ sạch sẽ hay không?",
        ],
        "ROOM_AMENITIES#COMFORT": [
            "cảm giác dễ chịu khi sử dụng các tiện nghi trong phòng",
            "Sử dụng các tiện nghi trong phòng có thoải mái hay không?",
        ],
        "ROOM_AMENITIES#DESIGN&FEATURES": [
            "tính năng và thiết kế của các tiện nghi trong phòng, như tivi, điều hòa, minibar",
            "Các tiện nghi trong phòng như tivi, điều hòa, minibar được thiết kế và trang bị ra sao?",
        ],
        "ROOM_AMENITIES#GENERAL": [
            "nhận xét chung về các tiện nghi trong phòng",
            "Nhìn chung, các tiện nghi trong phòng có tốt hay không?",
        ],
        "ROOM_AMENITIES#MISCELLANEOUS": [
            "các vấn đề khác về tiện nghi trong phòng",
            "Tiện nghi trong phòng còn điểm gì khác đáng chú ý?",
        ],
        "ROOM_AMENITIES#PRICES": [
            "chi phí phát sinh và giá cả liên quan đến tiện nghi trong phòng",
            "Sử dụng tiện nghi trong phòng có phát sinh chi phí cao hay không?",
        ],
        "ROOM_AMENITIES#QUALITY": [
            "tình trạng hoạt động và chất lượng của các tiện nghi trong phòng",
            "Các tiện nghi trong phòng có hoạt động tốt và chất lượng hay không?",
        ],
        "SERVICE#GENERAL": [
            "sự chuyên nghiệp, tốc độ và thái độ phục vụ của nhân viên",
            "Nhân viên khách sạn có phục vụ chuyên nghiệp, nhanh nhẹn và thái độ tốt hay không?",
        ],
    },
    "Phone": {
        "BATTERY": [
            "tốc độ sạc, độ bền và thời lượng sử dụng của pin điện thoại",
            "Pin điện thoại dùng có lâu, sạc nhanh và bền hay không?",
        ],
        "CAMERA": [
            "các tính năng của camera cùng chất lượng ảnh chụp, video quay",
            "Camera chụp ảnh, quay video có đẹp và nhiều tính năng hay không?",
        ],
        "DESIGN": [
            "vẻ ngoài, chất liệu và kiểu dáng của điện thoại",
            "Điện thoại có thiết kế, kiểu dáng và chất liệu đẹp hay không?",
        ],
        "FEATURES": [
            "phần mềm, chức năng và các tính năng của điện thoại",
            "Điện thoại có nhiều tính năng và phần mềm hữu ích hay không?",
        ],
        "GENERAL": [
            "trải nghiệm tổng thể và đánh giá chung về điện thoại",
            "Nhìn chung, trải nghiệm sử dụng điện thoại này có tốt hay không?",
        ],
        "PERFORMANCE": [
            "độ mượt, tốc độ xử lý và hiệu năng hoạt động của điện thoại",
            "Điện thoại chạy có mượt, nhanh và hiệu năng tốt hay không?",
        ],
        "PRICE": [
            "mức độ đáng tiền và giá cả của điện thoại",
            "Điện thoại này có giá cả hợp lý, xứng đáng với số tiền bỏ ra hay không?",
        ],
        "SCREEN": [
            "độ nhạy, kích thước và chất lượng hiển thị của màn hình",
            "Màn hình điện thoại hiển thị có đẹp, nhạy và kích thước phù hợp hay không?",
        ],
        "SER_ACC": [
            "chất lượng phụ kiện đi kèm hoặc thái độ phục vụ của người bán",
            "Người bán phục vụ có tốt, hoặc phụ kiện đi kèm có chất lượng hay không?",
        ],
        "STORAGE": [
            "khả năng lưu trữ và dung lượng bộ nhớ của điện thoại",
            "Điện thoại có đủ dung lượng lưu trữ và bộ nhớ hay không?",
        ],
    },
    "Education": {
        "Behavior": [
            "cách cư xử, sự tương tác và thái độ của giảng viên đối với sinh viên",
            "Giảng viên cư xử và tương tác với sinh viên có tốt hay không?",
        ],
        "Teaching Skill": [
            "phương pháp, cách truyền đạt và kỹ năng giảng bài của giảng viên",
            "Giảng viên có kỹ năng và phương pháp truyền đạt bài giảng tốt hay không?",
        ],
        "Suggestion": [
            "mong muốn cải thiện, góp ý và đề xuất từ phía sinh viên",
            "Sinh viên có những góp ý hay đề xuất cải thiện nào không?",
        ],
        "General": [
            "trải nghiệm tổng thể và đánh giá chung về giảng viên hoặc môn học",
            "Nhìn chung, giảng viên hoặc môn học này có tốt hay không?",
        ],
        "Exercise": [
            "tính hữu ích, mức độ và nội dung của bài tập được giao",
            "Bài tập được giao có nội dung phù hợp và hữu ích hay không?",
        ],
        "Knowledge": [
            "sự am hiểu và chiều sâu chuyên môn của giảng viên",
            "Giảng viên có kiến thức chuyên môn sâu và am hiểu hay không?",
        ],
        "Lecture Material": [
            "học liệu, giáo trình và tài liệu do giảng viên cung cấp",
            "Tài liệu và giáo trình giảng viên cung cấp có đầy đủ, chất lượng hay không?",
        ],
        "Equipment": [
            "cơ sở vật chất và công cụ hỗ trợ phục vụ giảng dạy",
            "Thiết bị và cơ sở vật chất phục vụ giảng dạy có đầy đủ hay không?",
        ],
        "Experience": [
            "khả năng liên hệ thực tiễn và kinh nghiệm thực tế của giảng viên",
            "Giảng viên có nhiều kinh nghiệm thực tế và liên hệ thực tiễn tốt hay không?",
        ],
        "Curriculum": [
            "cấu trúc, mức độ phù hợp và nội dung của chương trình giảng dạy",
            "Chương trình giảng dạy có nội dung và cấu trúc phù hợp hay không?",
        ],
        "Grading": [
            "tính công bằng và cách thức đánh giá, chấm điểm",
            "Việc chấm điểm và đánh giá có công bằng hay không?",
        ],
    },
    "Beauty": {
        "colour": [
            "màu sắc thực tế khi dùng có giống với mô tả hay không",
            "Sản phẩm lên màu có đúng và đẹp như mô tả hay không?",
        ],
        "others": [
            "những khía cạnh khác của sản phẩm không thuộc các mục cụ thể",
            "Sản phẩm còn điểm gì khác đáng chú ý ngoài những khía cạnh đã nêu?",
        ],
        "packing": [
            "cách đóng gói, hộp đựng và bao bì sản phẩm lúc giao hàng",
            "Sản phẩm được đóng gói và bao bì có cẩn thận hay không?",
        ],
        "price": [
            "mức độ đáng tiền và giá cả của sản phẩm làm đẹp",
            "Sản phẩm làm đẹp này có giá cả hợp lý, xứng đáng với số tiền bỏ ra hay không?",
        ],
        "shipping": [
            "chất lượng giao hàng và thời gian vận chuyển",
            "Vận chuyển và giao hàng có nhanh và tốt hay không?",
        ],
        "smell": [
            "hương thơm của sản phẩm khi sử dụng",
            "Sản phẩm có mùi hương dễ chịu khi dùng hay không?",
        ],
        "stayingpower": [
            "độ bám, thời gian lưu giữ hiệu quả và độ bền màu của sản phẩm",
            "Sản phẩm có giữ màu và hiệu quả lâu hay không?",
        ],
        "texture": [
            "độ mịn, cảm giác khi dùng và kết cấu của sản phẩm",
            "Kết cấu và cảm giác khi thoa sản phẩm có mịn màng, dễ chịu hay không?",
        ],
    },
}


def get_category_paraphrases(categories, domain):
    """Return {category_name_in_data: [paraphrase_1, paraphrase_2]} for every category.
    Mirrors mapper.get_category_descriptions()'s contract (same domain resolution, same
    category-name bridging via mapper._resolve_key), but for the paraphrase variants."""
    by_lower = {d.lower(): d for d in _DESCRIBED_DOMAINS}
    if domain.lower() not in by_lower:
        raise ValueError(f"unknown domain {domain!r}")
    domain = by_lower[domain.lower()]
    categories = list(categories)
    missing = [c for c in categories if not _resolve_key(domain, c)]
    if missing:
        raise ValueError(f"{domain}: no description for categories {missing}")
    return {c: CATEGORY_PARAPHRASES[domain][_resolve_key(domain, c)] for c in categories}
