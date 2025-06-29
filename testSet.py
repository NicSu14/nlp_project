import os
import shutil
import random
from tqdm import tqdm

def copy_sample_folders(src_dir, dest_dir, sample_count, label=""):
    folders = [f for f in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, f))]
    selected = random.sample(folders, min(sample_count, len(folders)))

    for folder_name in tqdm(selected, desc=f"Copying {label}", unit="folder"):
        src_path = os.path.join(src_dir, folder_name)
        dest_path = os.path.join(dest_dir, folder_name)
        shutil.copytree(src_path, dest_path)

    return len(selected)

def create_sample_dataset(dataset_root="dataset", sample_output="sample_dir", sample_size=500):
    phishing_src = os.path.join(dataset_root, "phishing")
    benign_src = os.path.join(dataset_root, "benign")
    sample_dir = os.path.join(dataset_root, sample_output)

    phishing_dest = os.path.join(sample_dir, "phishing")
    benign_dest = os.path.join(sample_dir, "benign")
    os.makedirs(phishing_dest, exist_ok=True)
    os.makedirs(benign_dest, exist_ok=True)

    print("📦 Sampling 500 folders each from phishing and benign...")

    phishing_copied = copy_sample_folders(phishing_src, phishing_dest, sample_size, label="Phishing")
    benign_copied = copy_sample_folders(benign_src, benign_dest, sample_size, label="Benign")

    print("\n✅ Sample extraction complete:")
    print(f"  Phishing samples copied: {phishing_copied}")
    print(f"  Benign samples copied:   {benign_copied}")
    print(f"  Output location:         {sample_dir}")

if __name__ == "__main__":
    create_sample_dataset()