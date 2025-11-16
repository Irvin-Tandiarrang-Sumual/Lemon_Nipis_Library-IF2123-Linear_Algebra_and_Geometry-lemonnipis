from typing import List, Dict, Tuple, Optional
import numpy as np
import os
import re

# 2.2.1 PERSIAPAN DATA DOKUMEN

def tokenize(text: str) -> List[str]:
    """
    Tokenisasi: memecah teks menjadi token/kata.
    Input:
        text: string mentah dari file dokumen
    Output:
        list of tokens (list[str])
    """
    raise NotImplementedError


def normalize_tokens(tokens: List[str]) -> List[str]:
    """
    Normalisasi:
    - Ubah huruf menjadi lowercase
    - Hilangkan karakter non-alphabet
    Output:
        list[str] (tokens normalisasi)
    """
    raise NotImplementedError


def stem_tokens(tokens: List[str]) -> List[str]:
    """
    Stemming: Mengubah kata-kata ke bentuk dasarnya (akar kata)
    """
    raise NotImplementedError


def remove_stopwords(tokens: List[str], stopwords: set) -> List[str]:
    """
    Menghapus stopword (kata yang tidak bermakna penting).
    Input:
        tokens: list of words
        stopwords: set berisi kata-kata yang dihapus
    Output:
        filtered tokens
    """
    raise NotImplementedError


def document_preprocess(text: str, stopwords: set, use_stemming=False) -> List[str]:
    """
    Pipeline preprocessing lengkap untuk 1 dokumen.
    Output:
        list kata siap digunakan untuk BoW dan TF-IDF
    """
    raise NotImplementedError

# 2.2.2 MATRIKS TERM-DOCUMENT

def build_vocabulary(preprocessed_docs: List[List[str]]) -> List[str]:
    """
    Menghasilkan vocabulary unik (list term) dari seluruh dokumen.
    Urutan bebas tapi harus konsisten.
    Output:
        vocab: list[str] ukuran m
    """
    raise NotImplementedError


def build_term_document_matrix(preprocessed_docs: List[List[str]], vocab: List[str]) -> np.ndarray:
    """
    Membangun matriks term-document A ukuran (m x n)
    A[i][j] = frekuensi term ke-i pada dokumen ke-j

    Output:
        A: numpy array shape (m, n)
    """
    raise NotImplementedError

# 2.2.3 PEMBOBOTAN TF-IDF

def compute_tf(A: np.ndarray) -> np.ndarray:
    """
    Hitung TF berdasarkan rumus:
        TF_ij = A_ij / sum(A_*,j)

    TF shape harus sama dengan A.
    """
    raise NotImplementedError


def compute_idf(A: np.ndarray) -> np.ndarray:
    """
    Hitung IDF:
        IDF_i = log10( n / (1 + df_i) )

    Output:
        idf_vector ukuran (m,)
    """
    raise NotImplementedError


def compute_tfidf(A: np.ndarray) -> np.ndarray:
    """
    Menghitung TF-IDF:
        TFIDF = diag(IDF) x TF

    Output:
        TFIDF matrix shape (m, n)
    """
    raise NotImplementedError

# 2.2.4 LSA Dengan SVD

def truncated_svd(A: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Melakukan truncated SVD:
    - Hitung AAT atau ATA (pilih yang lebih kecil)
    - Lakukan eigen decomposition
    - Ambil k singular values terbesar
    - Bangun U_k, Σ_k, V_k

    Output:
        U_k (m x k)
        Σ_k (k x k)
        V_k (n x k)
    """
    raise NotImplementedError

# 2.2.5 REPRESENTASI DOKUMEN DALAM RUANG SEMANTIK LATEN

def build_document_embeddings(V_k: np.ndarray, Sigma_k: np.ndarray) -> np.ndarray:
    """
    Embedding dokumen dihitung menggunakan:
        D = V_k x Σ_k

    Output:
        D shape (n x k)
        baris ke-i = embedding dokumen ke-i
    """
    raise NotImplementedError

# 2.2.6 METODE PERHITUNGAN SIMILARITAS

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Menormalisasi vector menjadi unit length.
    """
    raise NotImplementedError


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Menghitung cosine similarity:
        sim(a, b) = (a ⋅ b) / (||a|| ||b||)

    Output:
        nilai float antara -1 hingga 1
    """
    raise NotImplementedError


def compute_similarity_scores(query_vec: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
    """
    Menghitung similarity query terhadap seluruh dokumen.
    Output:
        array shape (n,), similarity scores
    """
    raise NotImplementedError

# HIGH LEVEL: BUILD MODEL + QUERY

def build_lsa_model(docs_path_list: List[str], stopwords: set, k: int = 50, use_stemming=False) -> Dict:
    """
    Build model untuk LSA:

    Langkah:
    1. Load dokumen -> preprocessing
    2. Build vocabulary
    3. Term-document matrix
    4. TF-IDF
    5. Truncated SVD
    6. Embedding dokumen

    Return dictionary berisi:
    - vocab
    - U_k, Σ_k, V_k
    - embeddings dokumen
    - preprocessed docs
    """
    raise NotImplementedError

def embed_query(query_text: str, model: Dict, stopwords: set, use_stemming=False) -> np.ndarray:
    """
    Memproses query teks agar sesuai pipeline dokumen:
    1. Preprocess query
    2. Ubah ke vector frekuensi
    3. TF-IDF (pakai IDF model, bukan IDF query)
    4. Project ke ruang laten:
        q_embed = (q_vec^T x U_k x Σ_k^{-1})

    Output:
        embedding query ukuran (k,)
    """
    raise NotImplementedError

def get_top_k_recommendations( query_text: str, model: Dict, stopwords: set, k: int = 5) -> List[Tuple[int, float]]:
    """
    Mendapatkan top-k dokumen paling mirip berdasarkan cosine similarity.

    Return list berisi tuple:
    (index_dokumen, similarity)
    """
    raise NotImplementedError