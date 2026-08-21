import re, string
from pyvi import ViTokenizer


def normalText(sent):
    #Chuẩn hóa tiếng Việt, xử lý emoj, chuẩn hóa tiếng Anh, thuật ngữ
    replace_list = {
        # Vietnamese diacritic normalization (combining-mark variants -> precomposed)
        'òa': 'oà',
        'óa': 'oá',
        'ỏa': 'oả',
        'õa': 'oã',
        'ọa': 'oạ',
        'òe': 'oè',
        'óe': 'oé',
        'ỏe': 'oẻ',
        'õe': 'oẽ',
        'ọe': 'oẹ',
        'ùy': 'uỳ',
        'úy': 'uý',
        'ủy': 'uỷ',
        'ũy': 'uỹ',
        'ụy': 'uỵ',
        'uả': 'ủa',
        'ả': 'ả',
        'ố': 'ố',
        'u´': 'ố',
        'ỗ': 'ỗ',
        'ồ': 'ồ',
        'ổ': 'ổ',
        'ấ': 'ấ',
        'ẫ': 'ẫ',
        'ẩ': 'ẩ',
        'ầ': 'ầ',
        'ỏ': 'ỏ',
        'ề': 'ề',
        'ễ': 'ễ',
        'ắ': 'ắ',
        'ủ': 'ủ',
        'ế': 'ế',
        'ở': 'ở',
        'ỉ': 'ỉ',
        'ẻ': 'ẻ',
        'àk': ' à ',
        'aˋ': 'à',
        'iˋ': 'ì',
        'ă´': 'ắ',
        'ử': 'ử',
        'e˜': 'ẽ',
        'y˜': 'ỹ',
        'a´': 'á',

        # Star-rating and misc symbol normalization (not sentiment-bearing)
        '⭐': 'star ',
        '*': 'star ',
        '🌟': 'star ',
        '😬': ' 😬 ',
        '😌': ' 😌 ',

        # Slang/typo/abbreviation -> canonical Vietnamese or English word
        '?': ' ? ',
        'ô kêi': ' ok ',
        'okie': ' ok ',
        ' o kê ': ' ok ',
        'okey': ' ok ',
        'ôkê': ' ok ',
        'oki': ' ok ',
        ' oke ': ' ok ',
        ' okay': ' ok ',
        'okê': ' ok ',
        ' tks ': ' cám ơn ',
        'thks': ' cám ơn ',
        'thanks': ' cám ơn ',
        'ths': ' cám ơn ',
        'thank': ' cám ơn ',
        'kg ': ' không ',
        'not': ' không ',
        ' kg ': ' không ',
        '"k ': ' không ',
        ' kh ': ' không ',
        'kô': ' không ',
        'hok': ' không ',
        ' kp ': ' không phải ',
        ' kô ': ' không ',
        '"ko ': ' không ',
        ' ko ': ' không ',
        ' k ': ' không ',
        'khong': ' không ',
        ' hok ': ' không ',
        'cute': ' dễ thương ',
        ' vs ': ' với ',
        'wa': ' quá ',
        'wá': ' quá',
        'j': ' gì ',
        '“': ' ',
        ' sz ': ' cỡ ',
        'size': ' cỡ ',
        ' đx ': ' được ',
        'dk': ' được ',
        'dc': ' được ',
        'đk': ' được ',
        'đc': ' được ',
        'authentic': ' chuẩn chính hãng ',
        ' aut ': ' chuẩn chính hãng ',
        ' auth ': ' chuẩn chính hãng ',
        'store': ' cửa hàng ',
        'shop': ' cửa hàng ',
        'sp': ' sản phẩm ',
        'gud': ' tốt ',
        'god': ' tốt ',
        'wel done': ' tốt ',
        'good': ' tốt ',
        'gút': ' tốt ',
        'sấu': ' xấu ',
        'gut': ' tốt ',
        ' tot ': ' tốt ',
        ' nice ': ' tốt ',
        'perfect': 'rất tốt',
        'bt': ' bình thường ',
        'time': ' thời gian ',
        'qá': ' quá ',
        ' ship ': ' giao hàng ',
        ' m ': ' mình ',
        ' mik ': ' mình ',
        'ể': 'ể',
        'product': 'sản phẩm',
        'quality': 'chất lượng',
        'chat': ' chất ',
        'excelent': 'hoàn hảo',
        'bad': 'tệ',
        'fresh': ' tươi ',
        'sad': ' tệ ',
        'date': ' hạn sử dụng ',
        'hsd': ' hạn sử dụng ',
        'quickly': ' nhanh ',
        'quick': ' nhanh ',
        'fast': ' nhanh ',
        'delivery': ' giao hàng ',
        ' síp ': ' giao hàng ',
        'beautiful': ' đẹp tuyệt vời ',
        ' tl ': ' trả lời ',
        ' r ': ' rồi ',
        ' shopE ': ' cửa hàng ',
        ' order ': ' đặt hàng ',
        'chất lg': ' chất lượng ',
        ' sd ': ' sử dụng ',
        ' dt ': ' điện thoại ',
        ' nt ': ' nhắn tin ',
        ' sài ': ' xài ',
        'bjo': ' bao giờ ',
        'thik': ' thích ',
        ' sop ': ' cửa hàng ',
        ' fb ': ' facebook ',
        ' face ': ' facebook ',
        ' very ': ' rất ',
        'quả ng ': ' quảng  ',
        'dep': ' đẹp ',
        ' xau ': ' xấu ',
        'delicious': ' ngon ',
        'hàg': ' hàng ',
        'qủa': ' quả ',
        'iu': ' yêu ',
        'fake': ' giả mạo ',
        'trl': 'trả lời',
        ' por ': ' tệ ',
        ' poor ': ' tệ ',
        'ib': ' nhắn tin ',
        'rep': ' trả lời ',
        'fback': ' feedback ',
        'fedback': ' feedback ',

        # Star-count normalization (below 3* -> 1*, above 3* -> 5*)
        '6 sao': ' 5star ',
        '6 star': ' 5star ',
        '5star': ' 5star ',
        '5 sao': ' 5star ',
        '5sao': ' 5star ',
        'starstarstarstarstar': ' 5star ',
        '1 sao': ' 1star ',
        '1sao': ' 1star ',
        '2 sao': ' 1star ',
        '2sao': ' 1star ',
        '2 starstar': ' 1star ',
        '1star': ' 1star ',
        '0 sao': ' 1star ',
        '0star': ' 1star ',
    }
    sent = sent.lower()
    for k, v in replace_list.items():
        # Bare alphabetic keys (no built-in surrounding space, e.g. 'sp', 'uả',
        # 'ib') are meant to match a standalone token or a whole malformed
        # vowel cluster -- plain .replace() also matches them as a substring
        # inside unrelated longer words (e.g. 'sp' inside "spa", 'uả' inside
        # "quả"/"quản"/"quảng"), silently corrupting real vocabulary. Anchor
        # these with \b so they only match at a real word boundary. Keys that
        # already carry their own leading/trailing space, or are punctuation/
        # emoji, keep the original substring replace (already low-risk).
        if k == k.strip() and k.isalpha():
            sent = re.sub(r'\b' + re.escape(k) + r'\b', v, sent)
        else:
            sent = sent.replace(k, v)


    sent = str(sent).replace('_',' ').replace('/',' trên ')
    sent = re.sub('-{2,}','',sent)
    sent = re.sub('\\s+',' ', sent)
    # Bugfix vs. upstream ABSA_LLMs/scripts/preprocessing.py: the original
    # (triệu|ngàn|trăm|k|K|) had no word boundary after k/K, so it could match
    # just the leading "k" of any following word (e.g. "không" -> "hông"),
    # silently destroying negation cues near numbers. \b anchors k/K so they
    # can only match a real standalone currency suffix, not a word prefix.
    patPrice = r'([0-9]+k?(\s?-\s?)[0-9]+\s?(k|K))|([0-9]+(.|,)?[0-9]+\s?(triệu|ngàn|trăm|k\b|K\b|))|([0-9]+(.[0-9]+)?Ä‘)|([0-9]+k)'
    patHagTag = r'#\s?[aăâbcdđeêghiklmnoôơpqrstuưvxyàằầbcdđèềghìklmnòồờpqrstùừvxỳáắấbcdđéếghíklmnóốớpqrstúứvxýảẳẩbcdđẻểghỉklmnỏổởpqrstủửvxỷạặậbcdđẹệghịklmnọộợpqrstụựvxỵãẵẫbcdđẽễghĩklmnõỗỡpqrstũữvxỹAĂÂBCDĐEÊGHIKLMNOÔƠPQRSTUƯVXYÀẰẦBCDĐÈỀGHÌKLMNÒỒỜPQRSTÙỪVXỲÁẮẤBCDĐÉẾGHÍKLMNÓỐỚPQRSTÚỨVXÝẠẶẬBCDĐẸỆGHỊKLMNỌỘỢPQRSTỤỰVXỴẢẲẨBCDĐẺỂGHỈKLMNỎỔỞPQRSTỦỬVXỶÃẴẪBCDĐẼỄGHĨKLMNÕỖỠPQRSTŨỮVXỸ]+'
    patURL = r"(?:http://|www.)[^\"]+"
    sent = re.sub(patURL,'website',sent)
    sent = re.sub(patHagTag,' hagtag ',sent)
    sent = re.sub(patPrice, ' giá tiền ', sent)
    sent = re.sub('\.+','.',sent)
    sent = re.sub('(hagtag\\s+)+',' hagtag ',sent)
    sent = re.sub('\\s+',' ',sent)
    return sent

def normalize_elonge_word(sent):
    s_new = ''
    for word in sent.split(' '):
        word_new = ''
        for char in word.strip():
            if char != word_new[-1]:
                word_new += char
    s_new += word_new.strip() + ' '
    return s_new

def tokenizer(text):
    token = ViTokenizer.tokenize(text)
    token = token.replace('giá tiền','giá_tiền').replace('Giá tiền','Giá_tiền')
    return token

def deleteIcon(text):
    text = text.lower()
    s = ''
    pattern = r"[a-zA-ZaăâbcdđeêghiklmnoôơpqrstuưvxyàằầbcdđèềghìklmnòồờpqrstùừvxỳáắấbcdđéếghíklmnóốớpqrstúứvxýảẳẩbcdđẻểghỉklmnỏổởpqrstủửvxỷạặậbcdđẹệghịklmnọộợpqrstụựvxỵãẵẫbcdđẽễghĩklmnõỗỡpqrstũữvxỹAĂÂBCDĐEÊGHIKLMNOÔƠPQRSTUƯVXYÀẰẦBCDĐÈỀGHÌKLMNÒỒỜPQRSTÙỪVXỲÁẮẤBCDĐÉẾGHÍKLMNÓỐỚPQRSTÚỨVXÝẠẶẬBCDĐẸỆGHỊKLMNỌỘỢPQRSTỤỰVXỴẢẲẨBCDĐẺỂGHỈKLMNỎỔỞPQRSTỦỬVXỶÃẴẪBCDĐẼỄGHĨKLMNÕỖỠPQRSTŨỮVXỸ,._]"
    
    for char in text:
        if char !=' ':
            if len(re.findall(pattern, char)) != 0:
                s+=char
            elif char == '_':
                s+=char
        else:
            s+=char
    s = re.sub('\\s+',' ',s)
    return s.strip()

def normalize_elonge_word(sent):
    """Collapse teencode-style elongation ('sooooo' -> 'so', 'ngonnnn' -> 'ngon').
    Only runs of 3+ identical characters are collapsed (to 1); runs of exactly 2
    are left alone. Native Vietnamese words never repeat a letter 3+ times, so a
    3+ run reliably signals artificial elongation -- but plenty of ordinary
    loanwords in this data (coffee, pizza, cheese, buffet, toffee, waffle) have
    a legitimate doubled letter, which the old threshold (collapsing ANY repeat,
    including exactly 2) was silently destroying into "cofe"/"piza"/"chese"/etc.
    """
    s_new = ''
    for word in sent.split(' '):
        word = word.strip()
        word_new = ''
        i = 0
        while i < len(word):
            j = i
            while j < len(word) and word[j] == word[i]:
                j += 1
            run_len = j - i
            word_new += word[i] if run_len >= 3 else word[i:j]
            i = j
        s_new += word_new + ' '
    return s_new.strip()

correct_mapping = {
    "ship": "vận chuyển",
    "shop": "cửa hàng",
    "m": "mình",
    "mik": "mình",
    "ko": "không",
    "k": " không ",
    "kh": "không",
    "khong": "không",
    "kg": "không",
    "khg": "không",
    "tl": "trả lời",
    "r": "rồi",
    "fb": "mạng xã hội", # facebook
    "face": "mạng xã hội",
    "thanks": "cảm ơn",
    "thank": "cảm ơn",
    "tks": "cảm ơn",
    "tk": "cảm ơn",
    "ok": "tốt",
    "dc": "được",
    "vs": "với",
    "đt": "điện thoại",
    "thjk": "thích",
    "qá": "quá",
    "trể": "trễ",
    "bgjo": "bao giờ",
    # Added from real Hotel_ABSA/Res_ABSA frequency analysis (frequent words
    # PhoBERT's tokenizer fragments into multiple BPE pieces), confirmed
    # against real sentence context, not guessed generically.
    "cũg": "cũng",
    "gía": "giá",
    "nhìu": "nhiều",
    "rùi": "rồi",
    "nhg": "nhưng",
    "trc": "trước",
    "cx": "cũng",
    "chổ": "chỗ",
    "sụa": "sữa",
    "dòn": "giòn",
    "cofe": "cà phê",
    "cmt": "bình luận",
    "nv": "nhân viên",
    "refil": "refill",
    "puding": "pudding",
    "bq": "bbq",       # "quán bq", "sốt bq" -- barbecue, not Vietnamese "k"=không sense
    "tsua": "trà sữa",  # confirmed by "vị trà khá rõ" context -- milk tea
    "bef": "beef",      # "bef steak top blade"
    "fre": "miễn phí",  # "giữ xe fre", "kem fre"
    "oder": "gọi",      # "mình oder trà sữa" = "I ordered milk tea"
    "tỡm": "tởm",       # typo for "tởm" (disgusting), as in "kinh tởm"
}
def tokmap(tok):
    if tok.lower() in correct_mapping:
        return correct_mapping[tok.lower()]
    else:
        return tok


def clean_doc(doc, word_segment=False, lower_case=False, max_length=512):
    for punc in string.punctuation:
        #doc = doc.replace(punc,' '+ punc + ' ')
        doc = doc.replace(punc," ")
    doc = normalText(doc)
    doc = deleteIcon(doc)
    # Removing multiple whitespaces
    doc = re.sub(r"\?", " \? ", doc)
    # Remove numbers
    doc = re.sub(r"[0-9]+", " num ", doc)
    # Split in tokens
    doc = re.sub('\\s+',' ',doc)
    doc = normalize_elonge_word(doc)
    if lower_case == True:
        doc = doc.lower()

    if word_segment == True:
        doc = tokenizer(doc)
        doc = doc.replace("giá _ tiền", "giá_tiền").replace("giátiền", "giá_tiền")
    else:
        doc = doc.replace("giá _ tiền", "giá tiền").replace("giátiền", "giá tiền")

    doc = re.sub('\\s+',' ',doc)
    doc = doc.strip()
    tokens = doc.split()
    tokens = map(tokmap, tokens)
    doc =  " ".join(tokens)
    doc = re.sub("\\s+", " ", doc).strip()
    array = doc.split(" ")
    if len(array) > max_length:
        aa = int(max_length/2) 
        doc = " ".join(array[:aa]) + " " + " ".join(array[-aa:])
    doc = re.sub("\\s+", " ", doc).strip()
    
    doc = doc.replace(". . .", " . ")
    return doc