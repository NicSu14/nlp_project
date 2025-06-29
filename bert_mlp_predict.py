import os
import ast
import numpy as np
import joblib
import torch
from transformers import MobileBertModel, MobileBertTokenizer
from bs4 import BeautifulSoup
from charset_normalizer import from_path
from extract_features_html import extract_features_phishing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from parser_for_training_no_patch import generate_text_representation

# ==== CONFIG ====
bert_dir = './model'
mlp_model_path = 'mlp_model_best_fold2.pkl'
sample_dir = 'dataset/sample_dir'  # Base sample_dir with phishing/benign subfolders

# ==== LOAD MODELS ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = MobileBertTokenizer.from_pretrained(bert_dir)
bert = MobileBertModel.from_pretrained(bert_dir).to(device)
bert.eval()
mlp = joblib.load(mlp_model_path)

# ==== READ FILE WITH FALLBACK ====
def read_file_with_fallback(path):
    results = from_path(path)
    encoding = results.best().encoding if results else None

    try:
        if encoding:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
    except UnicodeDecodeError:
        pass

    for fallback_enc in ["utf-16le", "utf-16be", "utf-16", "latin-1"]:
        try:
            with open(path, "r", encoding=fallback_enc) as f:
                print(f"Decoded {path} using fallback: {fallback_enc}")
                return f.read()
        except UnicodeDecodeError:
            continue

    print(f"❌ Failed to decode {path}")
    return None

# ==== PARSE URL FROM info.txt ====
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
                        if line.lower().startswith("url:"):
                            return line.split(":", 1)[1].strip()
                        if line.startswith("http"):
                            return line.strip()
        except (UnicodeDecodeError, UnicodeError):
            continue

    print(f"❌ Failed to decode or parse {info_path}")
    return None

# ==== EXTRACT FEATURES FROM SAMPLE FOLDER ====
def extract_features(folder_path):
    html_path = os.path.join(folder_path, 'html.txt')
    info_path = os.path.join(folder_path, 'info.txt')

    if not os.path.exists(html_path) or not os.path.exists(info_path):
        return None, None

    html = read_file_with_fallback(html_path)
    url = get_url_from_info(info_path)

    if not html or not url:
        return None, None

    soup = BeautifulSoup(html, 'html.parser')
    html_vec = extract_features_phishing(soup, url, feat_type='html')
    prettyHTML = generate_text_representation(html)

    tokens = tokenizer(prettyHTML, return_tensors='pt', truncation=True, max_length=256)
    with torch.no_grad():
        outputs = bert(**{k: v.to(device) for k, v in tokens.items()})
        cls_vec = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    final_vec = np.concatenate([cls_vec, html_vec])
    return final_vec, url

# ==== BATCH PREDICTION ====
def predict_on_samples():
    true_labels = []
    pred_labels = []
    failed_samples = 0

    for label in ["phishing", "benign"]:
        label_val = 1 if label == "phishing" else 0
        folder_path = os.path.join(sample_dir, label)
        if not os.path.isdir(folder_path):
            continue

        print(f"\n🔍 Predicting for: {label.upper()} samples")
        folders = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

        for sample_path in folders:
            try:
                print(f"→ Processing: {sample_path}")
                vec, url = extract_features(sample_path)
                if vec is not None:
                    pred = mlp.predict([vec])[0]
                    true_labels.append(label_val)
                    pred_labels.append(pred)
                    print(f"  [{label}] {url} → Prediction: {'Phishing' if pred == 1 else 'Benign'}")
                else:
                    print(f"  [{label}] {sample_path} → ❌ Could not extract features")
                    failed_samples += 1
            except Exception as e:
                print(f"  ⚠️ Error in {sample_path}: {e}")
                failed_samples += 1

    if true_labels:
        acc = accuracy_score(true_labels, pred_labels)
        prec = precision_score(true_labels, pred_labels)
        rec = recall_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels)

        print("\n📊 Final Evaluation Metrics:")
        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")
        print(f"\n⚠️ Failed to process: {failed_samples} samples")
    else:
        print("⚠️ No predictions were made.")

if __name__ == "__main__":
    predict_on_samples()