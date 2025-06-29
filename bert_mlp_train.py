import os
import numpy as np
import joblib
import torch
from transformers import MobileBertModel, MobileBertTokenizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from extract_features_html import extract_features_phishing
from charset_normalizer import from_path
from tqdm import tqdm
import random
import ast
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from parser_for_training_no_patch import generate_text_representation
from sklearn.model_selection import StratifiedKFold


# Paths
bert_dir = './model'
phish_dir = 'dataset/phishing'
benign_dir = 'dataset/benign'

# Load BERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = MobileBertTokenizer.from_pretrained(bert_dir)
bert = MobileBertModel.from_pretrained(bert_dir).to(device)
bert.eval()


def read_file_with_fallback(path):
    results = from_path(path)
    encoding = results.best().encoding if results else None

    try:
        if encoding:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
    except UnicodeDecodeError:
        pass  # fallback below

    # Fallback to utf-16 and utf-16le
    for fallback_enc in ["utf-16le", "utf-16be", "utf-16", "latin-1"]:
        try:
            with open(path, "r", encoding=fallback_enc) as f:
                print(f"Decoded using fallback: {fallback_enc}")
                return f.read()
        except UnicodeDecodeError:
            continue

    print("❌ Failed to decode file with all fallbacks.")
    return None

def get_url_from_info(info_path):
    if not os.path.exists(info_path):
        print(f"File does not exist: {info_path}")
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
                    # Fallback to legacy line parsing in case it's not a dict
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

def extract_features(sample_dir):
    html_path = os.path.join(sample_dir, 'html.txt')
    info_path = os.path.join(sample_dir, 'info.txt')
    if not os.path.exists(html_path) or not os.path.exists(info_path):
        print("DNE")
        return None, None, None

    html = read_file_with_fallback(html_path)
    #url = get_url_from_info(info_path)
    url = "https://example.com"
    if url is None:
        print(f"No URL found in {info_path}")
        return None, None, None
    if html is None or not html.strip():
        print(f"{html_path} is empty or unreadable.")
        return None, None, None

    soup = BeautifulSoup(html, 'html.parser')
    handcrafted_vec = extract_features_phishing(soup, url, feat_type='html')

    prettyHTML = generate_text_representation(html)
    # Get BERT CLS embedding
    try:
        tokens = tokenizer(prettyHTML, return_tensors='pt', truncation=True, max_length=256)
        with torch.no_grad():
            outputs = bert(**{k: v.to(device) for k, v in tokens.items()})
            cls_vec = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    except Exception as e:
        print(f"BERT encoding failed for {sample_dir}: {e}")
        return None, None, None

    # Concatenated vector
    final_vec = np.concatenate([cls_vec, handcrafted_vec])

    return cls_vec, handcrafted_vec, final_vec





def load_data(phish_dir, benign_dir, sample_size=2500, csv_output_path="features_output.csv"):
    X, y = [], []

    # Prepare output CSV
    with open(csv_output_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["CLS_Token", "Handcrafted_Feature_Vector", "Concatenated_Vector"])

        # Phishing samples
        print("Loading phishing samples...")
        phish_samples = [s for s in os.listdir(phish_dir) if os.path.isdir(os.path.join(phish_dir, s))]
        random.shuffle(phish_samples)
        phish_samples = phish_samples[:sample_size]

        for sampleid in tqdm(phish_samples):
            sample_dir = os.path.join(phish_dir, sampleid)
            cls_vec, html_vec, final_vec = extract_features(sample_dir)
            if final_vec is not None:
                X.append(final_vec)
                y.append(1)
                writer.writerow([
                    cls_vec.tolist(),
                    html_vec.tolist(),
                    final_vec.tolist()
                ])
            else:
                print("No final")

        # Benign samples
        print("Loading benign samples...")
        benign_samples = [s for s in os.listdir(benign_dir) if os.path.isdir(os.path.join(benign_dir, s))]
        random.shuffle(benign_samples)
        benign_samples = benign_samples[:sample_size]

        for sampleid in tqdm(benign_samples):
            sample_dir = os.path.join(benign_dir, sampleid)
            cls_vec, html_vec, final_vec = extract_features(sample_dir)
            if final_vec is not None:
                X.append(final_vec)
                y.append(0)
                writer.writerow([
                    cls_vec.tolist(),
                    html_vec.tolist(),
                    final_vec.tolist()
                ])
            else:
                print("No final")

    return np.array(X), np.array(y)
print("Loading data...")
X, y = load_data(phish_dir, benign_dir)
print(X)
print(y)
print("Starting 5-fold training...")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_f1 = -1.0
best_model = None
best_fold = -1

accuracies, precisions, recalls, f1s = [], [], [], []

for fold, (train_index, test_index) in enumerate(kf.split(X, y), start=1):
    print(f"\nFold {fold}...")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=200, random_state=fold)
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Fold {fold} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")

    # Track the best model by F1
    if f1 > best_f1:
        best_f1 = f1
        best_model = mlp
        best_fold = fold

    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1)

# Save the best model
if best_model:
    model_path = f"mlp_model_best_fold{best_fold}.pkl"
    joblib.dump(best_model, model_path)
    print(f"\nBest model saved from Fold {best_fold} to {model_path} (F1: {best_f1:.4f})")

# Final average results
print("\nAverage over 5 folds:")
print(f"Avg Accuracy: {np.mean(accuracies):.4f}")
print(f"Avg Precision: {np.mean(precisions):.4f}")
print(f"Avg Recall: {np.mean(recalls):.4f}")
print(f"Avg F1 Score: {np.mean(f1s):.4f}")