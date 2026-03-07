import os
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("Khong tim thay GEMINI_API_KEY!")

ROOT_FOLDER = r"D:\muvluvgg translate\muvluvgg-translation\translation"
MODEL_ID    = "gemini-2.0-flash-lite"
MAX_WORKERS = 10

client = genai.Client(api_key=API_KEY)

# ==========================================
# UTILS
# ==========================================

_CTRL  = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')
# Hiragana + Katakana — neu xuat hien trong OUTPUT = chua dich duoc
_JP    = re.compile(r'[\u3040-\u30ff]')
# Kanji pho bien — de detect input tieng Nhat
_KANJI = re.compile(r'[\u4e00-\u9fff]')

def sanitize(text):
    return _CTRL.sub('', str(text))

def has_japanese(text):
    """Kiem tra co Hiragana/Katakana khong."""
    return bool(_JP.search(str(text)))

def still_untranslated(val):
    """
    Kiem tra xem output co CHUA duoc dich khong.
    Output hop le (tieng Viet) KHONG duoc chua Hiragana/Katakana.
    """
    return has_japanese(val)

def repair_json(raw: str) -> str:
    """Sua cac loi JSON pho bien tu AI output."""
    raw = raw.strip()
    # Strip markdown fences
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    raw = raw.strip()
    # Fix invalid \uXXXX escapes
    raw = re.sub(r'\\u(?![0-9a-fA-F]{4})', '', raw)
    # Strip trailing commas
    raw = re.sub(r',(\s*[}\]])', r'\1', raw)
    return raw


# ==========================================
# PROMPTS
# ==========================================

PROMPT_SCENE = (
    "You are a professional Japanese-to-Vietnamese Visual Novel translator for Muv-Luv Girls Garden.\n\n"
    "STRICT OUTPUT RULES:\n"
    "- Return ONLY a valid JSON array of strings. NO markdown, NO extra text outside JSON.\n"
    "- Return EXACTLY N strings for N inputs. Do NOT merge, split, or skip any line.\n"
    "- Do NOT use double quotes (\") inside translated strings. Use single quotes or no quotes.\n"
    "- Do NOT add character names at the start of translations.\n"
    "- Output MUST be in Vietnamese. Do NOT output Japanese.\n\n"
    "TRANSLATION:\n"
    "- Natural Vietnamese, not word-for-word. Avoid passive voice.\n"
    "- Keep: ...... -- <r=...>...</r> %placeholder%\n\n"
    "PRONOUNS: Male lead (Ore)='Toi'. Girls to male='Em/Anh'. Girls together='To/Cau'. Enemy='Tao/May'.\n"
    "Keep Japanese suffixes: -san, -kun, -chan, -senpai\n"
    "TERMS: Chien Thuat Co | MG | Maze Shifter | Hoi Hoc Sinh | Ban Ky Luat\n"
)

PROMPT_SCENE_SINGLE = (
    "You are a Japanese-to-Vietnamese Visual Novel translator for Muv-Luv Girls Garden.\n"
    "Translate the ONE Japanese string in the input array into Vietnamese.\n"
    "OUTPUT: Exactly 1-element JSON array. Example: [\"Ban dich o day\"]\n"
    "- Output MUST be Vietnamese. NEVER output Japanese.\n"
    "- Do NOT use double quotes inside the string. Use single quotes if needed.\n"
    "- Keep: ...... -- <r=...>...</r> %placeholder%\n"
    "PRONOUNS: Ore='Toi'. Girls to male='Em/Anh'. Girls together='To/Cau'.\n"
    "TERMS: Chien Thuat Co | MG | Maze Shifter | Hoi Hoc Sinh | Ban Ky Luat\n"
)

PROMPT_TWEETY = (
    "You are a Japanese-to-Vietnamese translator for Muv-Luv Girls Garden (social media Tweety).\n\n"
    "STRICT OUTPUT RULES:\n"
    "- Return ONLY a valid JSON array of strings. NO extra text.\n"
    "- Return EXACTLY N strings for N inputs.\n"
    "- Do NOT use double quotes (\") inside strings.\n"
    "- Output MUST be Vietnamese. NEVER output Japanese.\n\n"
    "MANDATORY KEEP (do not translate these):\n"
    "- Token <<NL>> = in-game line break. Output MUST preserve <<NL>> exactly.\n"
    "- @mentions: @T_Commander, @Aoi_I, @Julia_B, @Flute_M, etc.\n"
    "- %usernameusernameuserna% and all %...% placeholders.\n"
    "- English team names: Sirius Sugar, Colorful Bouquet, Treble Quintet, etc.\n"
    "- Symbols: ~ (note) (heart) ! ?\n\n"
    "STYLE: Natural, youthful social-media tone.\n"
    "Girls use 'To/Minh'/'Cau'. With T_Commander: 'Em/Anh'.\n"
    "TERMS: MG | Maze | Chien Thuat Co | Hoi Hoc Sinh | Ban Ky Luat\n"
)

PROMPT_TWEETY_SINGLE = (
    "Translate ONE Japanese Tweety post to Vietnamese.\n"
    "OUTPUT: Exactly 1-element JSON array.\n"
    "- Output MUST be Vietnamese. NEVER output Japanese.\n"
    "- Do NOT use double quotes inside the string.\n"
    "- KEEP <<NL>> token exactly as-is (it is a line break).\n"
    "- KEEP @mentions, %placeholder%, English team names, symbols ~ ! ?\n"
    "- Girls: 'To/Minh'/'Cau'. With @T_Commander: 'Em/Anh'.\n"
    "TERMS: MG | Maze | Chien Thuat Co\n"
)

PROMPT_DICT = (
    "You are a Japanese-to-Vietnamese game translator for Muv-Luv Girls Garden.\n\n"
    "STRICT OUTPUT RULES:\n"
    "- Return ONLY a valid JSON array of strings. NO extra text.\n"
    "- Return EXACTLY N strings for N inputs.\n"
    "- Output MUST be Vietnamese.\n\n"
    "MANDATORY KEEP:\n"
    "- Rich Text tags: <color=#...> </material> </color>\n"
    "- Placeholders: %...%\n"
    "- English team names: Sirius Sugar, Colorful Bouquet, Treble Quintet, Chaos Maiden, Pre Class-A, Pyxis Ma Soeur, Inscarlet, Arcturus, Unity Edge, Trinity Jewel.\n"
    "- Empty string or '--' -> keep as-is.\n\n"
    "TRANSLATE: titles, chapter names, character descriptions in [...], NPC names.\n"
    "TERMS: Hoi Hoc Sinh | Ban Ky Luat | Chi huy\n"
)


# ==========================================
# API CALL — SMART RETRY 4 TANG
# ==========================================

def _call_once(text_list, prompt, temperature=0.3):
    clean   = [sanitize(t) for t in text_list]
    payload = json.dumps(clean, ensure_ascii=False)
    resp    = client.models.generate_content(
        model=MODEL_ID,
        contents=payload,
        config=types.GenerateContentConfig(
            system_instruction=prompt,
            response_mime_type="application/json",
            temperature=temperature,
        )
    )
    raw    = repair_json(resp.text)
    result = json.loads(raw)
    if not isinstance(result, list):
        raise ValueError(f"Not a list: {type(result)}")
    if len(result) != len(text_list):
        raise ValueError(f"Mismatch: In={len(text_list)} Out={len(result)}")
    # Kiem tra moi phan tu: neu con tieng Nhat = chua dich
    for i, val in enumerate(result):
        if still_untranslated(str(val)):
            raise ValueError(f"Untranslated at [{i}]: {repr(str(val)[:40])}")
    return result


def call_gemini(text_list, prompt, single_prompt=None):
    """
    Smart retry 4 tang:
      Tang 1-2 : Retry nguyen batch
      Tang 3   : Chia doi batch, goi de quy
      Tang 4   : Fallback tung dong voi single_prompt va nhiet do tang dan
    """
    if not text_list:
        return []

    # Tang 1 & 2
    for attempt in range(2):
        try:
            return _call_once(text_list, prompt)
        except Exception as e:
            print(f"    [!] Loi (lan {attempt+1}, {len(text_list)} dong): {e}")
            time.sleep(2)

    # Tang 3: chia doi
    if len(text_list) > 1:
        mid   = len(text_list) // 2
        print(f"    [~] Chia doi: {len(text_list)} -> {mid}+{len(text_list)-mid}")
        left  = call_gemini(text_list[:mid], prompt, single_prompt)
        right = call_gemini(text_list[mid:], prompt, single_prompt)
        if left is not None and right is not None:
            return left + right

    # Tang 4: tung dong mot voi nhiet do tang dan
    sp = single_prompt or prompt
    print(f"    [~] Fallback tung dong ({len(text_list)} dong)...")
    results = []
    for i, item in enumerate(text_list):
        translated = None
        temps = [0.3, 0.5, 0.7, 0.9]  # tang nhiet do moi lan retry
        for t_idx, temp in enumerate(temps):
            try:
                res = _call_once([item], sp, temperature=temp)
                val = res[0]
                # _call_once da validate roi, neu den day la OK
                translated = val
                break
            except Exception as e:
                if t_idx < len(temps) - 1:
                    time.sleep(1 + t_idx)
                else:
                    print(f"    [X] Giu nguyen dong {i}: {repr(str(item)[:50])}")
        results.append(translated if translated is not None else item)
    return results


# ==========================================
# PROCESSORS
# ==========================================

def process_scene_file(dir_path):
    source = os.path.join(dir_path, "zh_Hans.json")
    target = os.path.join(dir_path, "vi_VN.json")
    if os.path.exists(target):
        return

    try:
        with open(source, encoding='utf-8') as f:
            data = json.load(f)

        keys = list(data.keys())
        if not keys:
            return

        clean_keys = [k.replace('\\n', ' ').replace('\n', ' ') for k in keys]

        translated = []
        BATCH = 80
        for i in range(0, len(clean_keys), BATCH):
            batch = clean_keys[i:i + BATCH]
            res   = call_gemini(batch, PROMPT_SCENE, PROMPT_SCENE_SINGLE)
            translated.extend([v.replace('\\n', ' ').replace('\n', ' ') for v in res])

        out = {keys[i]: (translated[i] if i < len(translated) else keys[i])
               for i in range(len(keys))}

        with open(target, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=4)

        print(f"[Scenes] Xong: {os.path.basename(dir_path)} ({len(keys)} dong)")

    except Exception as e:
        print(f"[!] Loi Scenes {os.path.basename(dir_path)}: {e}")


def process_tweety_file(dir_path):
    source = os.path.join(dir_path, "zh_Hans.json")
    target = os.path.join(dir_path, "vi_VN.json")
    if os.path.exists(target):
        return

    print(f"[Tweety] Xu ly: {dir_path}")
    try:
        with open(source, encoding='utf-8') as f:
            data = json.load(f)

        NL = "<<NL>>"
        final_data = {}

        for category, sub_dict in data.items():
            if not isinstance(sub_dict, dict):
                final_data[category] = sub_dict
                continue

            orig_keys = list(sub_dict.keys())
            send_keys = [k.replace("\\n", NL) for k in orig_keys]

            trans_map = {}
            BATCH = 20
            for i in range(0, len(send_keys), BATCH):
                batch = send_keys[i:i + BATCH]
                res   = call_gemini(batch, PROMPT_TWEETY, PROMPT_TWEETY_SINGLE)
                for sk, tr in zip(batch, res):
                    # Kiem tra <<NL>> con nguyen khong
                    expected_nl = sk.count(NL)
                    if expected_nl > 0 and tr.count(NL) == 0:
                        print(f"    [~] <<NL>> bi mat, retry 1 dong...")
                        retry = call_gemini([sk], PROMPT_TWEETY_SINGLE, PROMPT_TWEETY_SINGLE)
                        tr = retry[0] if retry else sk
                    trans_map[sk] = tr.replace(NL, "\\n")

            final_data[category] = {
                ok: trans_map.get(sk, ok)
                for ok, sk in zip(orig_keys, send_keys)
            }

        with open(target, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)

        total = sum(len(v) for v in final_data.values() if isinstance(v, dict))
        print(f"[Tweety] Hoan tat! ({total} bai dang)")

    except Exception as e:
        print(f"[Tweety] Loi: {e}")


def process_nested_file(dir_path):
    source = os.path.join(dir_path, "zh_Hans.json")
    target = os.path.join(dir_path, "vi_VN.json")
    if os.path.exists(target):
        return

    folder = os.path.basename(dir_path)
    print(f"[{folder}] Xu ly: {dir_path}")

    try:
        with open(source, encoding='utf-8') as f:
            data = json.load(f)

        SKIP = {'', '--', '\uff1f\uff1f\uff1f', '\uff1f\uff1f\uff1f\uff1f', '???', '????'}

        all_keys = []
        for cat, sub in data.items():
            if isinstance(sub, dict):
                all_keys.extend(sub.keys())

        unique = list(set(k for k in all_keys if k not in SKIP and k.strip()))

        trans_map = {}
        BATCH = 80
        for i in range(0, len(unique), BATCH):
            batch = unique[i:i + BATCH]
            res   = call_gemini(batch, PROMPT_DICT)
            for orig, tr in zip(batch, res):
                trans_map[orig] = tr

        final_data = {}
        for cat, sub in data.items():
            if isinstance(sub, dict):
                final_data[cat] = {k: trans_map.get(k, k) for k in sub.keys()}
            else:
                final_data[cat] = sub

        with open(target, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)

        print(f"[{folder}] Hoan tat!")

    except Exception as e:
        print(f"[{folder}] Loi: {e}")


# ==========================================
# DISPATCH & MAIN
# ==========================================

def dispatch(dir_path):
    name = os.path.basename(dir_path).lower()
    if not os.path.exists(os.path.join(dir_path, "zh_Hans.json")):
        return
    if name == "tweety":
        process_tweety_file(dir_path)
    elif name in ("names", "titles"):
        process_nested_file(dir_path)
    else:
        process_scene_file(dir_path)


def main():
    print("=== TERTIA V12.0 — VALIDATE OUTPUT + TEMPERATURE ESCALATION ===")
    all_dirs = [x[0] for x in os.walk(ROOT_FOLDER)]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        ex.map(dispatch, all_dirs)
    print("\n=== HOAN TAT TOAN BO PROJECT ===")


if __name__ == "__main__":
    main()