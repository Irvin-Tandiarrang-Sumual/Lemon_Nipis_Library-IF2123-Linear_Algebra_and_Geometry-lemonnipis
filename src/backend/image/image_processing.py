from typing import List, Tuple, Callable, Optional, Dict
import numpy as np
from PIL import Image
import os
import pickle
import math
import time

# UTILITY FUNCTIONS (FILE I/O)

def get_image_paths(directory: str, exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> List[str]:
    """
    Mengambil seluruh path gambar di dalam folder dataset.
    Digunakan untuk membangun dataset sampul buku.

    TODO:
    - List seluruh file dalam directory
    - Pilih file dengan ekstensi sesuai parameter exts
    - Return list path lengkap (sorted)

    Return: List[str]
    """
    pass


def save_model(path: str, model: dict) -> None:
    """
    Menyimpan model PCA yang sudah diproses:
    (mean vector, eigenvector, projected dataset, dst.)

    TODO:
    - Gunakan pickle.dump untuk menyimpan dict model ke file binary.
    """
    pass


def load_model(path: str) -> dict:
    """
    Memuat model PCA yang telah disimpan sebelumnya.

    TODO:
    - Gunakan pickle.load untuk membuka file dan membaca dict
    """
    pass


# 2.1.1 PERSIAPAN DATA GAMBAR

def load_image_rgb(image_path: str) -> np.ndarray:
    """
    Load gambar dan ubah menjadi array numpy RGB float32 nilai 0--255.

    TODO:
    - Open image menggunakan PIL.Image
    - Convert ke RGB
    - Convert ke numpy array dtype float32
    """
    pass


def rgb_to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """
    Konversi RGB ke grayscale sesuai rumus spesifikasi:
    I = 0.2126 R + 0.7152 G + 0.0722 B

    Input  : array (H, W, 3)
    Output : array grayscale (H, W), dtype float32

    TODO:
    - Pisahkan channel R, G, B
    """
    pass


def resize_gray(gray: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resize gambar grayscale menjadi resolusi target_size x target_size.

    TODO:
    - Gunakan PIL.Image untuk resize
    - Return numpy array float32 0--255
    """
    pass


def flatten_image(gray_resized: np.ndarray) -> np.ndarray:
    """
    Mengubah gambar grayscale 2D menjadi vektor 1D (flatten).
    Vektor ini memiliki dimensi d = target_size * target_size.

    TODO:
    - Flatten (row-major)
    - Return numpy array float32
    """
    pass


def preprocess_image_to_vector(image_path: str, target_size: int = 100) -> np.ndarray:
    """
    Pipeline preprocessing untuk 1 gambar:
      1. Load RGB
      2. Konversi grayscale 
      3. Resize
      4. Normalisasi nilai piksel menjadi [0, 1]
      5. Flatten
    Return vektor shape (D,)

    TODO:
    - Panggil fungsi load_image_rgb -> rgb_to_grayscale -> resize_gray -> flatten
    - Bagi 255 untuk normalisasi
    """
    pass


def preprocess_dataset(image_paths: List[str], target_size: int = 100, progress_cb: Optional[Callable[[int], None]] = None) -> np.ndarray:
    """
    Preprocessing seluruh gambar dataset menjadi matriks X (N x D)
    sesuai spesifikasi TB:
      X[i] = vektor flatten gambar ke-i

    TODO:
    - Loop seluruh paths
    - Preprocess tiap gambar dengan preprocess_image_to_vector()
    - Isi matriks X
    - Jika progress_cb diberikan, update progress %
    """
    pass

# 2.1.2 NORMALISASI DATA

def compute_mean_vector(X: np.ndarray) -> np.ndarray:
    """
    Menghitung mean vector µ dari dataset:
    µ = (1/N) * Σ x_i

    Spec:
    - Hitung rata-rata per fitur (kolom)
    - Return vektor shape (D,)

    TODO:
    - Gunakan np.mean dengan axis=0
    """
    pass


def center_data(X: np.ndarray, mean_vector: np.ndarray) -> np.ndarray:
    """
    Melakukan centering data:
    x_i' = x_i - µ 

    Return X_centered shape (N, D)

    TODO:
    - Gunakan broadcasting
    """
    pass

#  2.1.3 PCA DENGAN SVD

def compute_pca_eig(X_centered: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Menghitung k komponen utama (principal components) dari data center X.

    Sesuai spesifikasi:
    - Jika N < D -> gunakan Gram matrix S = X X^T
    - Jika N >= D -> gunakan covariance C = X^T X
    - Lakukan eigen-decomposition
    - Pilih k eigenvector terbesar (nilai eigen terbesar)
    - Normalize setiap eigenvector ke unit length

    Return:
    - pcs: matriks eigenvector (D x k)
    - eigenvalues: array (k,)

    TODO:
    - Implementasi Gram trick untuk efisiensi
    - Implementasi eigen decomposition manual menggunakan np.linalg.eig
    - Sort descending berdasarkan eigenvalue
    - Normalisasi setiap eigenvector
    """
    pass


#  2.1.4 EIGEN-SAMPUL (Projection dan coefficients)

def project_dataset(X_centered: np.ndarray, pcs: np.ndarray) -> np.ndarray:
    """
    Menghitung c_i untuk setiap gambar dataset:
    c_i = U_k^T x_i'

    Return matriks (k x N)

    TODO:
    - Lakukan pcs.T @ X_centered.T
    """
    pass


def project_query(q_vect: np.ndarray, mean_vector: np.ndarray, pcs: np.ndarray) -> np.ndarray:
    """
    Proyeksi gambar kueri:
    q' = q - µ
    c_q = U_k^T q'

    TODO:
    - Center q_vect
    - Kalikan dengan pcs.T
    - Return vector shape (k,)
    """
    pass


#  2.1.5 METODE PERHITUNGAN SIMILARITAS

def compute_similarity_euclidean(projected_dataset: np.ndarray, projected_query: np.ndarray) -> np.ndarray:
    """
    Menghitung jarak Euclidean antara c_q (query) dan setiap c_i (dataset):
        d_i = || c_q - c_i ||_2

    Sesuai spesifikasi:
    - projected_dataset: shape (k, N)
    - projected_query: shape (k,)
    - k: jumlah eigenvector (dimensi ruang PCA)
    - N: jumlah gambar dalam dataset

    Return:
    - array jarak d_i (N,), semakin kecil semakin mirip

    TODO:
    - Hitung selisih: dataset - query
    - Hitung Euclidean distance
    """
    pass


#  HIGH-LEVEL MODEL BUILDING

def build_pca_model(dataset_dir: str, model_save_path: str, target_size: int = 100, k: int = 20, overwrite: bool = False, progress_cb: Optional[Callable[[int], None]] = None) -> Dict:
    """
    Membangun model PCA dari dataset sampul buku:
      1. Load seluruh gambar
      2. Preprocess -> X
      3. Compute mean and centered data
      4. PCA -> eigenvector
      5. Projection dataset
      6. Simpan model ke file

    TODO:
    - Implement semua langkah di atas
    - Gunakan semua fungsi sebelumnya
    """
    pass


def query_image_from_model(query_path: str, model: dict, top_n: int = 12, threshold: Optional[float] = None) -> List[Dict]:
    """
    Mencari gambar paling mirip dengan gambar kueri.

    TODO:
    - Preprocess query
    - Proyeksikan ke ruang eigen-sampul
    - Hitung similarity
    - Urutkan dari paling mirip
    - Return list berupa:
        { "path": ..., "score": ... }
    """
    pass


# MAIN (untuk testing)

if __name__ == "__main__":
    pass