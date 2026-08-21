import csv
import re


restaurant_dict = {
     'AMBIENCE#GENERAL': 'không gian',
     'DRINKS#PRICES': 'giá tiền đồ uống',
     'DRINKS#QUALITY': 'chất lượng đồ uống',
     'DRINKS#STYLE&OPTIONS': 'lựa chọn đồ uống',
     'FOOD#PRICES': 'giá tiền đồ ăn',
     'FOOD#QUALITY': 'chất lượng đồ ăn',
     'FOOD#STYLE&OPTIONS': 'lựa chọn đồ ăn',
     'LOCATION#GENERAL': 'địa chỉ',
     'RESTAURANT#GENERAL': 'nhà hàng nói chung',
     'RESTAURANT#MISCELLANEOUS': 'vấn đề khác',
     'RESTAURANT#PRICES': 'giá tiền nhà hàng',
     'SERVICE#GENERAL': 'dịch vụ'
}

phone_dict = {
     'BATTERY':'pin',
     'CAMERA':'máy ảnh',
     'DESIGN':'thiết kế',
     'FEATURES':'tính năng',
     'GENERAL':'nói chung',
     'PERFORMANCE':'hiệu suất',
     'PRICE':'giá tiền',
     'SCREEN':'màn hình',
     'SER_ACC':'phục vụ hoặc phụ kiện',
     'STORAGE':'bộ nhớ'
}


beauty_dict = {
    'colour': 'màu sắc',
    'others': 'vấn đề khác',
    'packing': 'bao bì',
    'price': 'giá tiền',
    'shipping': 'vận chuyển',
    'smell': 'mùi hương',
    'stayingpower': 'độ bền màu',
    'texture': 'kết cấu'
}


education_dict = {
    'Behavior': 'hành vi',              # Behavior
    'Curriculum': 'chương trình giảng dạy', # Curriculum
    'Equipment': 'thiết bị',            # Equipment
    'Exercise': 'bài tập',              # Exercise
    'Experience': 'kinh nghiệm',        # Experience
    'General': 'vấn đề chung',                 # General
    'Grading': 'chấm điểm',             # Grading
    'Knowledge': 'kiến thức',           # Knowledge
    'Lecture Material': 'tài liệu giảng dạy', # Lecture Material
    'Suggestion': 'đề xuất',            # Suggestion
    'Teaching Skill': 'kỹ năng giảng dạy' # Teaching Skill
}

technology_dict = {
    'Accessories': 'phụ kiện',       # Accessories
    'Configuration': 'cấu hình',     # Configuration
    'Genuineness': 'tính xác thực',  # Genuineness
    'Model': 'mẫu mã',                  # Model
    'Other': 'vấn đề khác',                 # Other
    'Performance': 'hiệu suất',      # Performance
    'Price': 'giá tiền',               # Price
    'Service': 'phục vụ',            # Service
    'Ship': 'vận chuyển'             # Ship
}


mother_dict = {
    'Genuineness': 'chân thật',  # Genuineness
    'Price': 'giá tiền',               # Price
    'Quality': 'chất lượng',         # Quality
    'Safety': 'an toàn',             # Safety
    'Service': 'phục vụ',            # Service
    'Ship': 'vận chuyển'             # Ship
}

hotel_dict = {
    'FACILITIES#CLEANLINESS': 'vệ sinh cơ sở vật chất',
    'FACILITIES#COMFORT': 'sự thoải mái cơ sở vật chất',
    'FACILITIES#DESIGN&FEATURES': 'thiết kế và tính năng cơ sở vật chất',
    'FACILITIES#GENERAL': 'cơ sở vật chất nói chung',
    'FACILITIES#MISCELLANEOUS': 'vấn đề khác cơ sở vật chất',
    'FACILITIES#PRICES': 'giá tiền cơ sở vật chất',
    'FACILITIES#QUALITY': 'chất lượng cơ sở vật chất',
    'FOOD&DRINKS#MISCELLANEOUS': 'vấn đề khác đồ ăn thức uống',
    'FOOD&DRINKS#PRICES': 'giá tiền đồ ăn thức uống',
    'FOOD&DRINKS#QUALITY': 'chất lượng đồ ăn thức uống',
    'FOOD&DRINKS#STYLE&OPTIONS': 'lựa chọn đồ ăn thức uống',
    'HOTEL#CLEANLINESS': 'vệ sinh khách sạn',
    'HOTEL#COMFORT': 'sự thoải mái khách sạn',
    'HOTEL#DESIGN&FEATURES': 'thiết kế và tính năng khách sạn',
    'HOTEL#GENERAL': 'khách sạn nói chung',
    'HOTEL#MISCELLANEOUS': 'vấn đề khác',
    'HOTEL#PRICES': 'giá tiền khách sạn',
    'HOTEL#QUALITY': 'chất lượng khách hàng',
    'LOCATION#GENERAL': 'địa chỉ',
    'ROOMS#CLEANLINESS': 'vệ sinh phòng',
    'ROOMS#COMFORT': 'sự thoải mái phòng',
    'ROOMS#DESIGN&FEATURES': 'thiết kế và tính năng phòng',
    'ROOMS#GENERAL': 'phòng nói chung',
    'ROOMS#MISCELLANEOUS': 'vấn đề khác của phòng',
    'ROOMS#PRICES': 'giá tiền phòng',
    'ROOMS#QUALITY': 'chất lượng phòng',
    'ROOM_AMENITIES#CLEANLINESS': 'vệ sinh tiện nghi phòng',
    'ROOM_AMENITIES#COMFORT': 'sự thoải mái tiện nghi phòng',
    'ROOM_AMENITIES#DESIGN&FEATURES': 'thiết kế và tính năng tiện nghi phòng',
    'ROOM_AMENITIES#GENERAL': 'tiện nghi phòng nói chung',
    'ROOM_AMENITIES#MISCELLANEOUS': 'vấn đề khác của tiện nghi phòng',
    'ROOM_AMENITIES#PRICES': 'giá tiền tiện nghi phòng',
    'ROOM_AMENITIES#QUALITY': 'chất lượng tiện nghi phòng',
    'SERVICE#GENERAL': 'phục vụ'
}


DOMAIN_DICTS = {
    "Restaurant": restaurant_dict,
    "Hotel": hotel_dict,
    "Mother": mother_dict,
    "Technology": technology_dict,
    "Education": education_dict,
    "Beauty": beauty_dict,
    "Phone": phone_dict,
}


def mapping_category(domain, category, lang_return='eng'):
    mapped_dict = DOMAIN_DICTS.get(domain)
    if mapped_dict is None:
        print("ERRORRR IN mapping_category FUNCTION: , ", domain)
        return None

    inversed_dict = {viet: eng for eng, viet in mapped_dict.items()}


    if lang_return == 'vie':
        return mapped_dict[category]
    else:
        return inversed_dict[category]


SENTIMENT_VIET2ENG = {
    'tốt': 'positive',
    'tệ': 'negative',
    'tạm': 'neutral'
}

SENTIMENT_ENG2VIET = {
    'positive': 'tốt',
    'negative': 'tệ',
    'neutral': 'tạm'
}


# -----------------------------------------------------------------------------
# Rich natural-language category descriptions, used as the semantic queries fed
# into the model's category-conditioned cross attention (train_mtl_acsa.py).
# These are intentionally fuller sentences than the *_dict short glosses above,
# which exist only for legacy label <-> label translation via mapping_category().
# Keys are each domain's canonical English category code, i.e. the keys of that
# domain's *_dict above (matching Restaurant/Hotel/Phone's native raw-data
# format, and recoverable for the other domains via mapping_category(..., 'eng')).
# -----------------------------------------------------------------------------
CATEGORY_DESCRIPTIONS = {
    "Restaurant": {
        "AMBIENCE#GENERAL": "không gian, bầu không khí, cách trang trí và sự thoải mái của quán",
        "DRINKS#PRICES": "giá cả và mức độ đắt rẻ của đồ uống",
        "DRINKS#QUALITY": "chất lượng, hương vị và độ ngon của đồ uống",
        "DRINKS#STYLE&OPTIONS": "loại đồ uống, cách pha chế, kích cỡ, topping và sự đa dạng lựa chọn",
        "FOOD#PRICES": "giá cả và mức độ đắt rẻ của món ăn",
        "FOOD#QUALITY": "chất lượng, hương vị, độ tươi ngon và cảm nhận về món ăn",
        "FOOD#STYLE&OPTIONS": "loại món ăn, cách chế biến, khẩu phần và sự đa dạng lựa chọn món",
        "LOCATION#GENERAL": "vị trí, địa điểm và mức độ dễ tìm của quán",
        "RESTAURANT#GENERAL": "đánh giá chung và trải nghiệm tổng thể về nhà hàng hoặc quán",
        "RESTAURANT#MISCELLANEOUS": "các tiện ích, giao hàng, giữ xe, vệ sinh, khuyến mãi và yếu tố khác của quán",
        "RESTAURANT#PRICES": "mức giá chung, hóa đơn và độ đáng tiền của nhà hàng hoặc quán",
        "SERVICE#GENERAL": "thái độ, tốc độ, sự chuyên nghiệp và chất lượng phục vụ của nhân viên",
    },
    "Hotel": {
        "FACILITIES#CLEANLINESS": "mức độ sạch sẽ, vệ sinh của các cơ sở vật chất và tiện ích chung của khách sạn",
        "FACILITIES#COMFORT": "sự thoải mái khi sử dụng các cơ sở vật chất và tiện ích chung",
        "FACILITIES#DESIGN&FEATURES": "thiết kế, kiến trúc và các tính năng của cơ sở vật chất và tiện ích chung",
        "FACILITIES#GENERAL": "đánh giá chung về cơ sở vật chất và tiện ích của khách sạn",
        "FACILITIES#MISCELLANEOUS": "các vấn đề khác liên quan đến cơ sở vật chất như wifi, thang máy, bãi đỗ xe",
        "FACILITIES#PRICES": "giá cả và mức phí sử dụng các cơ sở vật chất và tiện ích",
        "FACILITIES#QUALITY": "chất lượng và tình trạng hoạt động của các cơ sở vật chất và tiện ích",
        "FOOD&DRINKS#MISCELLANEOUS": "các vấn đề khác về đồ ăn thức uống như giờ phục vụ, cách sắp xếp buffet",
        "FOOD&DRINKS#PRICES": "giá cả và mức độ đắt rẻ của đồ ăn thức uống tại khách sạn",
        "FOOD&DRINKS#QUALITY": "chất lượng, hương vị và độ ngon của đồ ăn thức uống",
        "FOOD&DRINKS#STYLE&OPTIONS": "sự đa dạng, loại hình và cách chế biến đồ ăn thức uống",
        "HOTEL#CLEANLINESS": "mức độ sạch sẽ, vệ sinh chung của khách sạn",
        "HOTEL#COMFORT": "sự thoải mái, dễ chịu khi lưu trú tại khách sạn",
        "HOTEL#DESIGN&FEATURES": "thiết kế, kiến trúc và các tính năng nổi bật của khách sạn",
        "HOTEL#GENERAL": "đánh giá chung và trải nghiệm tổng thể về khách sạn",
        "HOTEL#MISCELLANEOUS": "các vấn đề khác của khách sạn không thuộc các khía cạnh cụ thể",
        "HOTEL#PRICES": "mức giá chung, hóa đơn và độ đáng tiền của khách sạn",
        "HOTEL#QUALITY": "chất lượng dịch vụ và trải nghiệm lưu trú tại khách sạn",
        "LOCATION#GENERAL": "vị trí, địa điểm và mức độ thuận tiện di chuyển của khách sạn",
        "ROOMS#CLEANLINESS": "mức độ sạch sẽ, vệ sinh của phòng nghỉ",
        "ROOMS#COMFORT": "sự thoải mái, dễ chịu khi nghỉ ngơi trong phòng",
        "ROOMS#DESIGN&FEATURES": "thiết kế, bài trí và các tính năng của phòng nghỉ",
        "ROOMS#GENERAL": "đánh giá chung về phòng nghỉ",
        "ROOMS#MISCELLANEOUS": "các vấn đề khác của phòng nghỉ không thuộc các khía cạnh cụ thể",
        "ROOMS#PRICES": "giá cả và mức phí của phòng nghỉ",
        "ROOMS#QUALITY": "chất lượng và tình trạng của phòng nghỉ",
        "ROOM_AMENITIES#CLEANLINESS": "mức độ sạch sẽ, vệ sinh của các tiện nghi trong phòng",
        "ROOM_AMENITIES#COMFORT": "sự thoải mái khi sử dụng các tiện nghi trong phòng",
        "ROOM_AMENITIES#DESIGN&FEATURES": "thiết kế và tính năng của các tiện nghi trong phòng như tivi, điều hòa, minibar",
        "ROOM_AMENITIES#GENERAL": "đánh giá chung về các tiện nghi trong phòng",
        "ROOM_AMENITIES#MISCELLANEOUS": "các vấn đề khác về tiện nghi trong phòng",
        "ROOM_AMENITIES#PRICES": "giá cả và chi phí phát sinh liên quan đến tiện nghi trong phòng",
        "ROOM_AMENITIES#QUALITY": "chất lượng và tình trạng hoạt động của các tiện nghi trong phòng",
        "SERVICE#GENERAL": "thái độ, tốc độ, sự chuyên nghiệp và chất lượng phục vụ của nhân viên",
    },
    "Phone": {
        "BATTERY": "thời lượng pin, tốc độ sạc và độ bền của pin điện thoại",
        "CAMERA": "chất lượng chụp ảnh, quay video và các tính năng của camera",
        "DESIGN": "kiểu dáng, chất liệu và vẻ ngoài thiết kế của điện thoại",
        "FEATURES": "các tính năng, chức năng và phần mềm của điện thoại",
        "GENERAL": "đánh giá chung và trải nghiệm tổng thể về điện thoại",
        "PERFORMANCE": "hiệu năng xử lý, độ mượt và tốc độ hoạt động của điện thoại",
        "PRICE": "giá cả và mức độ đáng tiền của điện thoại",
        "SCREEN": "chất lượng hiển thị, độ nhạy và kích thước màn hình",
        "SER_ACC": "thái độ phục vụ của người bán hoặc chất lượng phụ kiện đi kèm",
        "STORAGE": "dung lượng bộ nhớ và khả năng lưu trữ của điện thoại",
    },
    "Education": {
        "Behavior": "thái độ, cách cư xử và sự tương tác của giảng viên với sinh viên",
        "Teaching Skill": "kỹ năng, phương pháp và cách truyền đạt bài giảng của giảng viên",
        "Suggestion": "các đề xuất, góp ý và mong muốn cải thiện từ sinh viên",
        "General": "đánh giá chung và trải nghiệm tổng thể về giảng viên hoặc môn học",
        "Exercise": "nội dung, mức độ và tính hữu ích của bài tập được giao",
        "Knowledge": "chiều sâu kiến thức chuyên môn và sự am hiểu của giảng viên",
        "Lecture Material": "tài liệu, giáo trình và học liệu được giảng viên cung cấp",
        "Equipment": "thiết bị, công cụ hỗ trợ và cơ sở vật chất phục vụ giảng dạy",
        "Experience": "kinh nghiệm thực tế và khả năng liên hệ thực tiễn của giảng viên",
        "Curriculum": "nội dung, cấu trúc và mức độ phù hợp của chương trình giảng dạy",
        "Grading": "cách thức chấm điểm, đánh giá và tính công bằng trong chấm điểm",
    },
    "Beauty": {
        "colour": "màu sắc của sản phẩm khi sử dụng, có đúng và đẹp như mô tả hay không",
        "others": "các vấn đề khác không thuộc những khía cạnh cụ thể của sản phẩm",
        "packing": "bao bì, hộp đựng và cách đóng gói sản phẩm khi giao hàng",
        "price": "giá cả và mức độ đáng tiền của sản phẩm làm đẹp",
        "shipping": "thời gian và chất lượng vận chuyển, giao hàng",
        "smell": "mùi hương của sản phẩm khi sử dụng",
        "stayingpower": "độ bền màu, độ bám và thời gian lưu giữ hiệu quả của sản phẩm",
        "texture": "kết cấu, độ mịn và cảm giác khi sử dụng sản phẩm",
    },
    "Mother": {
        "Genuineness": "tính xác thực, độ chính hãng của sản phẩm mẹ và bé",
        "Price": "giá cả và mức độ đáng tiền của sản phẩm",
        "Quality": "chất lượng và độ an tâm khi sử dụng sản phẩm",
        "Safety": "độ an toàn của sản phẩm đối với trẻ nhỏ",
        "Service": "thái độ phục vụ, tư vấn và chăm sóc khách hàng của người bán",
        "Ship": "thời gian và chất lượng vận chuyển, giao hàng",
    },
    "Technology": {
        "Accessories": "phụ kiện đi kèm và mức độ đầy đủ của phụ kiện",
        "Configuration": "cấu hình, thông số kỹ thuật của sản phẩm",
        "Genuineness": "tính xác thực, độ chính hãng của sản phẩm",
        "Model": "kiểu dáng, mẫu mã và thiết kế bên ngoài của sản phẩm",
        "Other": "các vấn đề khác không thuộc những khía cạnh cụ thể",
        "Performance": "hiệu năng xử lý, độ mượt và hiệu suất hoạt động của sản phẩm",
        "Price": "giá cả và mức độ đáng tiền của sản phẩm công nghệ",
        "Service": "thái độ phục vụ, tư vấn và chăm sóc khách hàng của người bán",
        "Ship": "thời gian và chất lượng vận chuyển, giao hàng",
    },
}


# -----------------------------------------------------------------------------
# Loader for the cleaned ABSA_LLMs Pair-format data (data/Pair/<Domain>/{Train,
# Dev,Test}.csv), used as the source for every domain except Restaurant, which
# still trains from its own original Res_ABSA/*.txt files.
#
# Two label encodings show up in that data:
#   - Restaurant/Phone keep an exact 'raw_output' column of {CATEGORY, sentiment}
#     pairs already in each domain's canonical English category codes.
#   - Hotel/Education/Beauty/Mother/Technology only have 'output', a flattened
#     Vietnamese sentence built by joining "<dict value> <sentiment word>"
#     segments with " và ". Naively splitting on " và " is unsafe because some
#     category phrases contain the literal word "và" themselves (e.g. Hotel's
#     "thiết kế và tính năng phòng"), so parse_pair_output anchors on the full
#     set of known "<phrase> <sentiment>" strings for that domain instead of
#     splitting blindly. Verified to parse 100% of Hotel/Education/Beauty/
#     Mother/Technology rows in data/Pair with zero leftovers.
# -----------------------------------------------------------------------------
_RAW_OUTPUT_RE = re.compile(r"\{([^,{}]+),\s*([^{}]+)\}")


def parse_pair_output(domain, output):
    """Recover [(english_category, sentiment_en), ...] from an 'output' string.
    Returns (labels, leftover); leftover is non-empty only if a segment near
    the end of the string didn't match any known category+sentiment phrase."""
    domain_dict = DOMAIN_DICTS[domain]
    candidates = []
    for eng_key, vie_phrase in domain_dict.items():
        for vie_sent, sent_en in SENTIMENT_VIET2ENG.items():
            candidates.append((f"{vie_phrase} {vie_sent}", eng_key, sent_en))
    candidates.sort(key=lambda c: -len(c[0]))  # longest phrase first

    labels = []
    remaining = output.strip()
    while remaining:
        if remaining.startswith("và "):
            remaining = remaining[3:].strip()
        match = next(
            (c for c in candidates if remaining == c[0] or remaining.startswith(c[0] + " và ")),
            None,
        )
        if match is None:
            break
        cand_str, eng_key, sent_en = match
        labels.append((eng_key, sent_en))
        remaining = remaining[len(cand_str):].strip()
    return labels, remaining


def load_pair_examples(domain, csv_path):
    """Returns (examples, skipped) where examples is a list of (sample_id,
    text, labels) tuples, skipped counts rows with empty text or labels that
    couldn't be fully recovered from 'output' (none observed in practice)."""
    examples = []
    skipped = 0
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            text = (row.get("input") or "").strip()
            if not text:
                skipped += 1
                continue

            if row.get("raw_output"):
                labels = [
                    (cat.strip(), sent.strip().lower())
                    for cat, sent in _RAW_OUTPUT_RE.findall(row["raw_output"])
                ]
            else:
                labels, leftover = parse_pair_output(domain, row.get("output") or "")
                if leftover:
                    skipped += 1
                    continue

            if not labels:
                skipped += 1
                continue
            examples.append((str(i), text, labels))
    return examples, skipped
