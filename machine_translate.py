import os
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. .envからAPIキーを読み込む
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("[!] LỖI: Không tìm thấy GEMINI_API_KEY. Hãy kiểm tra lại file .env!")

# --- 構成設定 ---
ROOT_FOLDER = r"D:\muvluvgg translate\muvluvgg-translation\translation"
MODEL_ID = "gemini-2.0-flash-lite"
MAX_WORKERS = 10
MAX_RETRIES = 3

client = genai.Client(api_key=API_KEY)

# ==========================================
# PROMPTS
# ==========================================

# Dành cho scene files (Nhật → Việt)
PROMPT_SCENE = (
    "Bạn là dịch giả Visual Novel chuyên nghiệp (Nhật → Việt) cho game 'Muv-Luv Girls Garden' (Bối cảnh: Học viện Quân sự / Mecha Sci-fi).\n\n"

    "=== OUTPUT FORMAT (TUYỆT ĐỐI TUÂN THỦ) ===\n"
    "• Trả về ĐÚNG định dạng JSON Array chứa các chuỗi String. KHÔNG trả về định dạng [N] hay thêm text ngoài JSON.\n"
    "• Trả về đúng N phần tử cho N input. Không thêm, không bớt, không gộp câu.\n"
    "• KHÔNG dùng dấu nháy kép (\") bên trong câu thoại để tránh lỗi JSON. Dùng nháy đơn (') hoặc nháy Nhật (「」) nếu cần.\n"
    "• KHÔNG đặt tên nhân vật vào đầu bản dịch.\n\n"

    "=== CHẤT LƯỢNG DỊCH — ƯU TIÊN SỐ 1 ===\n"
    "• Dịch như người Việt viết, không phải như máy dịch. Nếu dịch sát chữ nghe cứng → dịch thoát ý, giữ cảm xúc gốc.\n"
    "• Tránh lạm dụng từ 'đã/được/bị', câu bị động, hoặc từ Hán-Việt nặng nề (Cấm dùng: Tại hạ, Tiểu nữ, Hỗn đản).\n"
    "• Ưu tiên động từ có hình ảnh. Cấm tuyệt đối cấu trúc dịch 'một cách + [tính từ]'.\n\n"

    "=== XƯNG HÔ (MUV-LUV) ===\n"
    "• Nam chính (Ore/俺): BẮT BUỘC dịch là 'Tôi'. KHÔNG DÙNG 'Ta' hay 'Tớ'.\n"
    "• Nữ sinh với Nam chính: Xưng 'Em' - Gọi 'Anh' (hoặc 'Cậu/Mình' nếu bạn bè).\n"
    "• Nữ sinh với nhau: Xưng 'Tớ/Mình' - Gọi 'Cậu'.\n"
    "• Kẻ địch / Xung đột: Xưng 'Tao' - Gọi 'Mày' hoặc 'Tôi - Cô'.\n"
    "• Giữ nguyên hậu tố Nhật (-san, -kun, -chan, -senpai) nếu có.\n\n"

    "=== CẢM XÚC & NHỊP ĐIỆU ===\n"
    "• …… (6 dấu chấm) → Giữ nguyên '……'. Không rút ngắn thành '...'.\n"
    "• —— (Gạch ngang dài) → Giữ nguyên để thể hiện câu bị cắt ngang.\n"
    "• Cảm thán: ああ → À…/Ồ… | くそ → Chết tiệt/Khốn kiếp | まあ → Thôi thì…/Chà…\n\n"

    "=== THUẬT NGỮ CỐ ĐỊNH ===\n"
    "• 戦術機 → Chiến Thuật Cơ | MG → MG | メイズシフター → Maze Shifter\n"
    "• 生徒会 → Hội Học Sinh | 風紀委員 → Ban Kỷ Luật\n"
)

# Dành cho tweety (Trung → Việt, phong cách mạng xã hội)
PROMPT_TWEETY = (
    "Bạn là dịch giả game chuyên nghiệp (Trung → Việt) cho 'Muv-Luv Girls Garden'.\n\n"

    "=== OUTPUT FORMAT (TUYỆT ĐỐI TUÂN THỦ) ===\n"
    "• Trả về ĐÚNG định dạng JSON Array chứa các chuỗi String. KHÔNG thêm text ngoài JSON.\n"
    "• Trả về đúng N phần tử cho N input. Không thêm, không bớt.\n"
    "• KHÔNG dùng dấu nháy kép (\") bên trong câu để tránh lỗi JSON.\n\n"

    "=== QUY TẮC BẮT BUỘC ===\n"
    "• GIỮ NGUYÊN '\\\\n' — đây là ký tự xuống dòng trong game, KHÔNG xoá hay thay thế.\n"
    "• GIỮ NGUYÊN các @mention (ví dụ: ＠T_Commander, ＠Aoi_I).\n"
    "• GIỮ NGUYÊN %usernameusernameuserna% và mọi placeholder dạng %...%.\n"
    "• GIỮ NGUYÊN tên nhân vật/đội tiếng Anh: Sirius Sugar, Colorful Bouquet, Treble Quintet, v.v.\n\n"

    "=== PHONG CÁCH DỊCH ===\n"
    "• Dịch tự nhiên, trẻ trung theo phong cách mạng xã hội (Twitter/Tweety).\n"
    "• Giữ cảm xúc, biểu cảm gốc: ~ ♪ ♥ ！ ？ các ký hiệu đặc biệt → GIỮ NGUYÊN.\n"
    "• Xưng hô nữ sinh với nhau: 'Tớ/Mình' - 'Cậu'. Với chỉ huy (T_Commander): 'Em/Anh'.\n\n"

    "=== THUẬT NGỮ CỐ ĐỊNH ===\n"
    "• MG → MG | メイズ/迷宫 → Maze | 戦術機/战术机甲 → Chiến Thuật Cơ\n"
    "• 生徒会/学生会 → Hội Học Sinh | 风纪委员 → Ban Kỷ Luật\n"
)

# Dành cho titles/names (Trung → Việt, danh từ/tên riêng)
PROMPT_DICT = (
    "Bạn là biên dịch viên game (Trung → Việt) cho 'Muv-Luv Girls Garden'.\n\n"

    "=== OUTPUT FORMAT (TUYỆT ĐỐI TUÂN THỦ) ===\n"
    "• Trả về ĐÚNG định dạng JSON Array chứa các chuỗi String.\n"
    "• Trả về đúng N phần tử cho N input. Không thêm, không bớt.\n\n"

    "=== QUY TẮC BẮT BUỘC ===\n"
    "• GIỮ NGUYÊN thẻ Rich Text: <color=#...>, </material>, </color> và nội dung bên trong nếu là tên tiếng Anh.\n"
    "• GIỮ NGUYÊN %usernameusernameuserna% và mọi placeholder dạng %...%.\n"
    "• GIỮ NGUYÊN '\\\\r\\\\n' ở cuối chuỗi nếu có.\n"
    "• GIỮ NGUYÊN tên đội/nhóm tiếng Anh: Sirius Sugar, Colorful Bouquet, Treble Quintet, Chaos Maiden, Pre Class-A, Pyxis Ma Soeur, Inscarlet, Treble Quintet, Arcturus, Unity Edge, Trinity Jewel.\n"
    "• GIỮ NGUYÊN tên riêng tiếng Nhật đã phiên âm: các tên như Meru, Naniro, Shirona, Rami, Chiyuru, Aoi, Uruu, Julia, Suiran, Saya, Fii, Anisu, v.v.\n\n"

    "=== QUY TẮC DỊCH ===\n"
    "• Danh từ chung/tiêu đề: Dịch sang tiếng Việt tự nhiên (VD: '第1节第1话' → 'Tiết 1 Tập 1', '序章第1话' → 'Mở đầu Tập 1').\n"
    "• Subtitle/mô tả nhân vật: Dịch nghĩa, giữ sắc thái (VD: '[笨拙又纯真的未来王牌]' → '[Cô gái vụng về nhưng thuần khiết, Át chủ bài tương lai]').\n"
    "• Tên NPC/nhân vật phụ: Dịch mô tả sang tiếng Việt (VD: '声音沙哑的教师' → 'Giáo viên giọng khàn khàn').\n"
    "• 学生会 → Hội Học Sinh | 风纪委员 → Ban Kỷ Luật | 指挥官 → Chỉ huy\n"
    "• Chuỗi rỗng '' → Giữ nguyên chuỗi rỗng ''.\n"
    "• '--' → Giữ nguyên '--'.\n"
)


# ==========================================
# CORE LOGIC
# ==========================================

def call_gemini(text_list, prompt):
    """Gọi API với kiểm tra số lượng phần tử trả về."""
    if not text_list:
        return []
    payload = json.dumps(text_list, ensure_ascii=False)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=payload,
                config=types.GenerateContentConfig(
                    system_instruction=prompt,
                    response_mime_type="application/json",
                    temperature=0.5,
                )
            )
            result = json.loads(response.text)
            if len(result) != len(text_list):
                raise ValueError(f"Mismatch: In {len(text_list)} vs Out {len(result)}")
            return result
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"    [!] Lỗi API: {e}")
                return None
            time.sleep(1)
    return None


def process_scene_file(dir_path):
    """Xử lý scene files: xoá newline trước khi gửi API."""
    source = os.path.join(dir_path, "zh_Hans.json")
    target = os.path.join(dir_path, "vi_VN.json")

    if os.path.exists(target):
        return

    try:
        with open(source, 'r', encoding='utf-8') as f:
            data = json.load(f)

        keys_to_translate = list(data.keys())
        if not keys_to_translate:
            return

        # Xoá \n trước khi gửi AI (Unity không hỗ trợ newline trong scene)
        cleaned_texts = []
        for k in keys_to_translate:
            clean_k = k.replace('\\n', ' ').replace('\n', ' ')
            cleaned_texts.append(clean_k)

        translated_vals = []
        batch_size = 200
        batches = [cleaned_texts[i:i + batch_size] for i in range(0, len(cleaned_texts), batch_size)]

        for batch in batches:
            res = call_gemini(batch, PROMPT_SCENE)
            if res:
                clean_res = [v.replace('\\n', ' ').replace('\n', ' ') for v in res]
                translated_vals.extend(clean_res)
            else:
                translated_vals.extend(batch)

        target_data = {}
        for i, original_key in enumerate(keys_to_translate):
            target_data[original_key] = translated_vals[i] if i < len(translated_vals) else original_key

        with open(target, 'w', encoding='utf-8') as f:
            json.dump(target_data, f, ensure_ascii=False, indent=4)

        print(f"[Scenes] Xong: {os.path.basename(dir_path)} ({len(keys_to_translate)} dòng)")

    except Exception as e:
        print(f"[!] Lỗi tại {dir_path}: {e}")


def process_tweety_file(dir_path):
    """
    Xử lý tweety/zh_Hans.json.
    Cấu trúc: { "tweetyPosts": { japanese_key: chinese_value, ... } }
    - Dịch KEY tiếng Nhật → Việt, đặt làm value trong vi_VN.json
    - GIỮ NGUYÊN \\n trong key (xuống dòng bài đăng mạng xã hội)
    - GIỮ NGUYÊN @mention và %placeholder%
    """
    source = os.path.join(dir_path, "zh_Hans.json")
    target = os.path.join(dir_path, "vi_VN.json")

    if os.path.exists(target):
        return

    print(f"[Tweety] Đang xử lý: {dir_path}")
    try:
        with open(source, 'r', encoding='utf-8') as f:
            data = json.load(f)

        final_data = {}

        for category, sub_dict in data.items():
            if not isinstance(sub_dict, dict):
                final_data[category] = sub_dict
                continue

            # Dịch từ KEY tiếng Nhật sang Việt
            all_keys = list(sub_dict.keys())

            trans_map = {}
            batch_size = 80
            batches = [all_keys[i:i + batch_size] for i in range(0, len(all_keys), batch_size)]

            for batch in batches:
                res = call_gemini(batch, PROMPT_TWEETY)
                if res:
                    for origin, trans in zip(batch, res):
                        trans_map[origin] = trans
                else:
                    for origin in batch:
                        trans_map[origin] = origin

            # Key gốc (Nhật) giữ nguyên, value = bản dịch Việt
            translated_sub = {k: trans_map.get(k, k) for k in all_keys}
            final_data[category] = translated_sub

        with open(target, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)

        total = sum(len(v) for v in final_data.values() if isinstance(v, dict))
        print(f"[Tweety] Hoàn tất! ({total} bài đăng)")

    except Exception as e:
        print(f"[Tweety] Lỗi: {e}")


def process_nested_file(dir_path):
    """
    Xử lý titles/zh_Hans.json và names/zh_Hans.json.
    Cấu trúc: { "category": { japanese_key: chinese_value, ... }, ... }
    - Dịch KEY tiếng Nhật → Việt, đặt làm value trong vi_VN.json
    - GIỮ NGUYÊN Rich Text tags trong key nếu có
    - Bỏ qua chuỗi đặc biệt không cần dịch: '', '--', '？？？'
    """
    source = os.path.join(dir_path, "zh_Hans.json")
    target = os.path.join(dir_path, "vi_VN.json")

    if os.path.exists(target):
        return

    folder_name = os.path.basename(dir_path)
    print(f"[{folder_name}] Đang xử lý: {dir_path}")

    try:
        with open(source, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Thu thập tất cả KEYS cần dịch, loại bỏ trùng lặp
        SKIP_KEYS = {'', '--', '？？？', '？？？？'}
        all_keys = []
        for category, sub_dict in data.items():
            if isinstance(sub_dict, dict):
                all_keys.extend(sub_dict.keys())
            else:
                all_keys.append(category)

        unique_keys = list(set(k for k in all_keys if k not in SKIP_KEYS and k.strip()))

        trans_map = {}
        batch_size = 100
        batches = [unique_keys[i:i + batch_size] for i in range(0, len(unique_keys), batch_size)]

        for batch in batches:
            res = call_gemini(batch, PROMPT_DICT)
            if res:
                for origin, trans in zip(batch, res):
                    trans_map[origin] = trans
            else:
                for origin in batch:
                    trans_map[origin] = origin

        # Ghép lại cấu trúc gốc: key Nhật giữ nguyên, value = bản dịch Việt
        final_data = {}
        for category, sub_dict in data.items():
            if isinstance(sub_dict, dict):
                final_data[category] = {k: trans_map.get(k, k) for k in sub_dict.keys()}
            else:
                final_data[category] = sub_dict

        with open(target, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)

        print(f"[{folder_name}] Hoàn tất!")

    except Exception as e:
        print(f"[{folder_name}] Lỗi: {e}")


def dispatch(dir_path):
    """Phân loại folder và gọi hàm xử lý phù hợp."""
    folder_name = os.path.basename(dir_path).lower()

    if folder_name == "tweety":
        if os.path.exists(os.path.join(dir_path, "zh_Hans.json")):
            process_tweety_file(dir_path)
    elif folder_name in ["names", "titles"]:
        if os.path.exists(os.path.join(dir_path, "zh_Hans.json")):
            process_nested_file(dir_path)
    else:
        if os.path.exists(os.path.join(dir_path, "zh_Hans.json")):
            process_scene_file(dir_path)


def main():
    print("=== TERTIA V9.0 (TWEETY + TITLES + NAMES OPTIMIZED) ===")
    print("Môi trường: Đã tải GEMINI_API_KEY từ .env")

    all_dirs = [x[0] for x in os.walk(ROOT_FOLDER)]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(dispatch, all_dirs)

    print("\n=== HOÀN TẤT TOÀN BỘ PROJECT ===")


if __name__ == "__main__":
    main()