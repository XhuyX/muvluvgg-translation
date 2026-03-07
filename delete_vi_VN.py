import os

ROOT_FOLDER = r"D:\muvluvgg translate\muvluvgg-translation\translation"

def main():
    deleted = 0
    skipped = 0

    for dirpath, _, filenames in os.walk(ROOT_FOLDER):
        for filename in filenames:
            if filename == "vi_VN.json":
                full_path = os.path.join(dirpath, filename)
                try:
                    os.remove(full_path)
                    print(f"[DEL] {full_path}")
                    deleted += 1
                except Exception as e:
                    print(f"[ERR] Không thể xoá {full_path}: {e}")
                    skipped += 1

    print(f"\n=== HOÀN TẤT ===")
    print(f"Đã xoá : {deleted} file(s)")
    print(f"Lỗi   : {skipped} file(s)")

if __name__ == "__main__":
    main()
