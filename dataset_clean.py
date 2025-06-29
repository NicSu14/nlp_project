import os
import shutil
import ast
from charset_normalizer import from_path
from tqdm import tqdm

def get_url_from_info(info_path):
    if not os.path.exists(info_path):
        return None

    encodings_to_try = []

    results = from_path(info_path)
    if results and results.best():
        encodings_to_try.append(results.best().encoding)

    encodings_to_try += ["utf-8", "utf-16le", "utf-16be", "utf-16", "latin-1"]

    for enc in encodings_to_try:
        try:
            with open(info_path, "r", encoding=enc) as f:
                content = f.read().strip()
                if not content:
                    return None
                try:
                    data = ast.literal_eval(content)
                    return data.get("url", None)
                except (ValueError, SyntaxError):
                    f.seek(0)
                    for line in f:
                        line = line.strip()
                        if line.lower().startswith("url:"):
                            return line.split(":", 1)[1].strip()
                        if line.startswith("http"):
                            return line.strip()
        except (UnicodeDecodeError, UnicodeError):
            continue
    return None

def should_delete(folder_path):
    try:
        files = os.listdir(folder_path)
        if "html.txt" not in files or "info.txt" not in files:
            return True
        info_path = os.path.join(folder_path, "info.txt")
        url = get_url_from_info(info_path)
        return not url
    except Exception:
        return True

def clean_dataset(root_path):
    for category in ["phishing", "benign"]:
        base_path = os.path.join(root_path, category)
        if not os.path.exists(base_path):
            print(f"Skipping missing directory: {base_path}")
            continue

        folders = [os.path.join(base_path, f) for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

        total = len(folders)
        deleted = 0

        for folder_path in tqdm(folders, desc=f"Cleaning {category}", unit="folder"):
            if should_delete(folder_path):
                shutil.rmtree(folder_path)
                deleted += 1

        kept = total - deleted
        print(f"\nSummary for '{category}':")
        print(f"  Total folders:  {total}")
        print(f"  Deleted:        {deleted}")
        print(f"  Kept:           {kept}\n")


def clean_folder(folder_path):
    deleted_count = 0
    untouched = True

    try:
        for filename in os.listdir(folder_path):
            if filename not in ["html.txt", "info.txt"]:
                path = os.path.join(folder_path, filename)
                if os.path.isfile(path):
                    os.remove(path)
                    deleted_count += 1
                    untouched = False
                elif os.path.isdir(path):
                    shutil.rmtree(path)
                    deleted_count += 1
                    untouched = False
    except Exception as e:
        print(f"Error cleaning {folder_path}: {e}")

    return deleted_count, untouched

def clean_all_folders(root_path):
    total_folders = 0
    folders_untouched = 0
    total_files_deleted = 0

    for category in ["phishing", "benign"]:
        base_path = os.path.join(root_path, category)
        if not os.path.exists(base_path):
            continue

        folders = [os.path.join(base_path, f) for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        total_folders += len(folders)

        for folder_path in tqdm(folders, desc=f"Cleaning {category}", unit="folder"):
            deleted, untouched = clean_folder(folder_path)
            total_files_deleted += deleted
            if untouched:
                folders_untouched += 1

    print("\n🧾 Final Report:")
    print(f"  Total folders processed:        {total_folders}")
    print(f"  Folders untouched (no cleanup): {folders_untouched}")
    print(f"  Total files deleted:            {total_files_deleted}")

if __name__ == "__main__":
    clean_all_folders("dataset")
