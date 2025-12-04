from typing import List, Tuple, Callable, Optional, Dict
import numpy as np
from PIL import Image
import os
import pickle
import time

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

def power_iteration(A: np.ndarray, num_iter: int = 1000, tol: float = 1e-9) -> Tuple[float, np.ndarray]:
    n = A.shape[0]
    v = np.random.randn(n).astype(np.float64)
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-15:
        v = np.ones(n, dtype=np.float64)
        v_norm = np.linalg.norm(v)
    v /= v_norm
    last_val = 0.0
    eigen_val = 0.0
    for _ in range(num_iter):
        w = np.dot(A, v)
        norm_w = np.linalg.norm(w)
        if norm_w < 1e-15:
            break
        v = w / norm_w
        eigen_val = float(np.dot(v, np.dot(A, v)))
        if abs(eigen_val - last_val) < tol:
            break
        last_val = eigen_val
    return eigen_val, v

def top_k_eigen_power_iteration(A: np.ndarray, k: int, num_iter: int = 1000, tol: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    A_copy = A.copy().astype(np.float64)
    eigenvalues = []
    eigenvectors = []
    for i in range(k):
        val, vec = power_iteration(A_copy, num_iter=num_iter, tol=tol)
        eigenvalues.append(val)
        eigenvectors.append(vec.copy())
        A_copy = A_copy - val * np.outer(vec, vec)
    eigvals_arr = np.array(eigenvalues, dtype=np.float64)
    eigvecs_mat = np.column_stack(eigenvectors) 
    return eigvals_arr, eigvecs_mat

def compute_pca_svd(X_centered: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    N, d = X_centered.shape
    L = np.dot(X_centered.astype(np.float64), X_centered.T.astype(np.float64)) / N
    scale_factor = np.max(np.abs(L))
    if scale_factor == 0:
        scale_factor = 1.0
    L_scaled = L / scale_factor
    eig_val_scaled, eig_vec_small = top_k_eigen_power_iteration(L_scaled, k)
    eig_val = eig_val_scaled * scale_factor
    idx = np.argsort(eig_val)[::-1]
    eig_val = eig_val[idx]
    eig_vec_small = eig_vec_small[:, idx]
    eig_val[eig_val < 0] = 0.0
    singular_values = np.sqrt(eig_val * N)
    eigenfaces_unnormalized = np.dot(X_centered.T.astype(np.float64), eig_vec_small)
    eigenfaces = np.zeros_like(eigenfaces_unnormalized)
    for i in range(min(k, eigenfaces_unnormalized.shape[1])):
        sigma = singular_values[i]
        if sigma > 1e-10:
            eigenfaces[:, i] = eigenfaces_unnormalized[:, i] / sigma
        else:
            eigenfaces[:, i] = 0.0
    return eigenfaces.T.astype(np.float32), eig_val.astype(np.float32)

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

query_file = "src/backend/image/tc1.png" 
results = query_image_from_model(query_file, pca_model, top_n=10)

print("Hasil Pencarian:")
for res in results:
    print(f"- {res['file_name']} (Jarak: {res['score']:.4f})")
