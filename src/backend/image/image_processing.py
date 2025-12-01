from typing import List, Tuple, Callable, Optional, Dict
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import pickle
import math
import time
import glob

# UTILITY FUNCTIONS (FILE I/O)

def get_image_paths(directory: str, exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> List[str]:
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Direktori dataset tidak ditemukan: {directory}")
        
    image_paths = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(exts):
                image_paths.append(os.path.join(root, file))
    
    image_paths.sort()
    return image_paths

def save_model(path: str, model: dict) -> None:
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
        
    try:
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model berhasil disimpan ke: {path}")
    except Exception as e:
        print(f"Gagal menyimpan model: {e}")

def load_model(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File model tidak ditemukan: {path}")
        
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model berhasil dimuat dari: {path}")
        return model
    except Exception as e:
        raise RuntimeError(f"Gagal memuat model: {e}")

# 2.1.1 PERSIAPAN DATA GAMBAR

def load_image_rgb(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    return np.asarray(img, dtype=np.float32)

def rgb_to_grayscale(rgb: np.ndarray) -> np.ndarray:
    R = rgb[..., 0]
    G = rgb[..., 1]
    B = rgb[..., 2]
    gray = 0.2126*R + 0.7152*G + 0.0722*B
    return gray.astype(np.float32)

def resize_gray(gray: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    img = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8))
    img_resized = img.resize((target_width, target_height), Image.Resampling.BILINEAR)
    return np.asarray(img_resized, dtype=np.float32)

def flatten_image(gray_resized: np.ndarray) -> np.ndarray:
    return gray_resized.flatten(order='F').astype(np.float32)

def preprocess_image_data_to_vector(image_path: str) -> np.ndarray:
    rgb = load_image_rgb(image_path)
    gray = rgb_to_grayscale(rgb)
    vect = flatten_image(gray)
    return vect

def preprocess_dataset(image_paths: List[str]) -> np.ndarray:
    dataset_vectors = []
    for f in image_paths:
        vect = preprocess_image_data_to_vector(f)
        dataset_vectors.append(vect)
    return np.array(dataset_vectors, dtype=np.float32)
        
# 2.1.2 NORMALISASI DATA

def compute_mean_vector(X: np.ndarray) -> np.ndarray:
    mean_vector = np.mean(X, axis=0)
    return mean_vector

def center_data(X: np.ndarray, mean_vector: np.ndarray) -> np.ndarray:
    X_centered = X - mean_vector
    return X_centered

#  2.1.3 PCA DENGAN SVD

def qr_decomposition_gram_schmidt(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n, m = A.shape
    Q = np.zeros((n, m), dtype=np.float64) 
    R = np.zeros((m, m), dtype=np.float64)

    for j in range(m):
        v = A[:, j].copy()
        
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], v)
            v = v - R[i, j] * Q[:, i]
            
        R[j, j] = np.linalg.norm(v)
        
        if R[j, j] > 1e-14:
            Q[:, j] = v / R[j, j]
        else:
            Q[:, j] = 0 
            
    return Q, R

def _wilkinson_shift(A_sub: np.ndarray) -> float:
    m = A_sub.shape[0]
    if m < 2:
        return A_sub[-1, -1]
    
    a = A_sub[m-2, m-2]
    b = A_sub[m-2, m-1] 
    c = A_sub[m-1, m-1] 
    
    delta = (a - c) / 2.0
    sign_delta = 1.0 if delta >= 0 else -1.0
    
    denom = delta + sign_delta * np.sqrt(delta**2 + b**2)
    
    if abs(denom) < 1e-14: 
        mu = c
    else:
        mu = c - (b**2) / denom
        
    return mu

def find_eigen_qr_algorithm(A: np.ndarray, max_iter_per_eig: int = 1000, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    A_copy = A.astype(np.float64).copy()
    n = A_copy.shape[0]
    V = np.eye(n, dtype=np.float64)
    
    for m in range(n, 1, -1):
        iter_count = 0
        while iter_count < max_iter_per_eig:
            iter_count += 1
            
            if abs(A_copy[m-1, m-2]) < tol:
                A_copy[m-1, m-2] = 0.0
                A_copy[m-2, m-1] = 0.0
                break
            
            mu = _wilkinson_shift(A_copy[:m, :m])
            
            mu_perturbed = mu + 1e-10 
            
            A_shifted = A_copy[:m, :m] - mu_perturbed * np.eye(m)
            Q_sub, R_sub = qr_decomposition_gram_schmidt(A_shifted)
            
            A_copy[:m, :m] = np.dot(R_sub, Q_sub) + mu_perturbed * np.eye(m)
            
            V[:, :m] = np.dot(V[:, :m], Q_sub)
            
    return np.diagonal(A_copy), V

def compute_pca_svd(X_centered: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    N, d = X_centered.shape
    
    L = np.dot(X_centered.astype(np.float64), X_centered.T.astype(np.float64)) / N
    
    scale_factor = np.max(np.abs(L))
    L_scaled = L / scale_factor
    
    eig_val_scaled, eig_vec_small = find_eigen_qr_algorithm(L_scaled)
    
    eig_val = eig_val_scaled * scale_factor
    
    idx = np.argsort(eig_val)[::-1]
    eig_val = eig_val[idx]
    eig_vec_small = eig_vec_small[:, idx]
    
    if k > N: k = N
    eig_val_k = eig_val[:k]
    eig_vec_small_k = eig_vec_small[:, :k]
    
    eig_val_k[eig_val_k < 0] = 0 
    singular_values = np.sqrt(eig_val_k * N)
    
    # 5. Proyeksi ke Eigenfaces (d x k)
    # Gunakan X_centered float64 untuk presisi
    eigenfaces_unnormalized = np.dot(X_centered.T.astype(np.float64), eig_vec_small_k)
    
    # 6. Normalisasi dengan Sigma
    eigenfaces = np.zeros_like(eigenfaces_unnormalized)
    for i in range(k):
        sigma = singular_values[i]
        if sigma > 1e-10:
            eigenfaces[:, i] = eigenfaces_unnormalized[:, i] / sigma
        else:
            eigenfaces[:, i] = 0
            
    return eigenfaces.T.astype(np.float32), eig_val_k.astype(np.float32)

#  2.1.4 EIGEN-SAMPUL (Projection dan coefficients)

def project_dataset(X_centered: np.ndarray, pcs: np.ndarray) -> np.ndarray:
    X_32 = X_centered.astype(np.float32)
    pcs_32 = pcs.astype(np.float32)
    projected_data = np.dot(X_32, pcs_32.T)
    return projected_data

def project_query(q_vect: np.ndarray, mean_vector: np.ndarray, pcs: np.ndarray) -> np.ndarray:
    q_vect_32 = q_vect.astype(np.float32)
    mean_32 = mean_vector.astype(np.float32)
    pcs_32 = pcs.astype(np.float32)
    q_centered = q_vect_32 - mean_32
    q_weights = np.dot(pcs_32, q_centered)
    return q_weights

#  2.1.5 METODE PERHITUNGAN SIMILARITAS

def compute_similarity_euclidean(projected_dataset: np.ndarray, projected_query: np.ndarray) -> np.ndarray:
    dataset_32 = projected_dataset.astype(np.float32)
    query_32 = projected_query.astype(np.float32)
    diff = dataset_32 - query_32
    distances = np.linalg.norm(diff, axis=1)
    return distances

#  HIGH-LEVEL MODEL BUILDING

def _preprocess_single_image_dynamic(path: str, w: int, h: int) -> np.ndarray:
    rgb = load_image_rgb(path)
    gray = rgb_to_grayscale(rgb)
    resized = resize_gray(gray, w, h)
    return flatten_image(resized)

def build_pca_model(dataset_dir: str, model_save_path: str, target_width: int = 200, target_height: int = 300, k: int = 50, overwrite: bool = False, progress_cb: Optional[Callable[[int], None]] = None) -> Dict:
    if os.path.exists(model_save_path) and not overwrite:
        print("Model sudah ada.")
        return load_model(model_save_path)

    print(f"Memulai Training PCA... (Target Size: {target_width}x{target_height}, k={k})")
    start_time = time.time()

    image_paths = get_image_paths(dataset_dir)
    N = len(image_paths)
    if N == 0:
        raise ValueError("Dataset kosong! Tidak ada gambar ditemukan.")
    print(f"Ditemukan {N} gambar.")

    print("Memproses gambar...")
    data_list = []
    for i, path in enumerate(image_paths):
        vec = _preprocess_single_image_dynamic(path, target_width, target_height)
        data_list.append(vec)
        if progress_cb: progress_cb(i)
    
    X = np.array(data_list, dtype=np.float32)
    
    print("Menghitung Mean & Centering...")
    mean_vector = compute_mean_vector(X)
    X_centered = center_data(X, mean_vector)

    print("Menghitung Eigenfaces (QR Algorithm)...")
    eigenfaces, eigenvalues = compute_pca_svd(X_centered, k)

    print("Memproyeksikan dataset...")
    projected_data = project_dataset(X_centered, eigenfaces)

    model = {
        "mean_vector": mean_vector,
        "eigenfaces": eigenfaces,
        "eigenvalues": eigenvalues,
        "projected_dataset": projected_data,
        "image_paths": image_paths, 
        "image_size": (target_width, target_height), 
        "k": k
    }

    save_model(model_save_path, model)
    
    elapsed = time.time() - start_time
    print(f"Training Selesai dalam {elapsed:.2f} detik.")
    
    return model

def query_image_from_model(query_path: str, model: dict, top_n: int = 12, threshold: Optional[float] = None) -> List[Dict]:
    mean_vec = model["mean_vector"]
    pcs = model["eigenfaces"]
    db_weights = model["projected_dataset"]
    db_paths = model["image_paths"]
    w, h = model["image_size"]
    
    try:
        q_vec = _preprocess_single_image_dynamic(query_path, w, h)
    except Exception as e:
        print(f"Error loading query image: {e}")
        return []

    q_weight = project_query(q_vec, mean_vec, pcs)

    dists = compute_similarity_euclidean(db_weights, q_weight)

    sorted_indices = np.argsort(dists)
    
    results = []
    for idx in sorted_indices:
        score = dists[idx]
        
        if threshold is not None and score > threshold:
            break 
            
        result_item = {
            "index": int(idx),
            "file_path": db_paths[idx],
            "file_name": os.path.basename(db_paths[idx]),
            "score": float(score) 
        }
        results.append(result_item)
        
        if len(results) >= top_n:
            break
            
    return results

if __name__ == "__main__":
    # MAIN (untuk testing)
    DATASET_DIR = "data/covers"
    MODEL_PATH = "data/model_pca.pkl"

    pca_model = build_pca_model(
        dataset_dir=DATASET_DIR,
        model_save_path=MODEL_PATH,
        target_width=200,
        target_height=300,
        k=50
    )

    query_file = "data/covers/2488.jpg"
    results = query_image_from_model(query_file, pca_model, top_n=5)

    print("Hasil Pencarian:")
    for res in results:
        print(f"- {res['index']} {res['file_path']} {res['file_name']} (Jarak: {res['score']:.4f})")
