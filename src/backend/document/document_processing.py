from typing import List, Dict, Tuple, Optional
import numpy as np
import os
import time
import re
import math
import random
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pickle
import multiprocessing as mp
nltk.download('stopwords')
porter_stemmer = PorterStemmer()
token_regex = re.compile(r"[A-Za-z]+")
# 2.2.1 PERSIAPAN DATA DOKUMEN

def tokenize(text: str) -> List[str]: ####AMAN
    """
    Tokenisasi: memecah teks menjadi token/kata.
    Input:
        text: string mentah dari file dokumen
    Output:
        list of tokens (list[str])
    """
    return token_regex.findall(text)

def normalize_tokens(tokens: List[str]) -> List[str]: ## dari baca harusnya aman
    """
    Normalisasi:
    - Ubah huruf menjadi lowercase
    - Hilangkan karakter non-alphabet
    Output:
        list[str] (tokens normalisasi)
    """
    return [t.lower() for t in tokens if t and t.isalpha()]

def stem_tokens(tokens: List[str]) -> List[str]: ## aman
    """
    Stemming: Mengubah kata-kata ke bentuk dasarnya (akar kata)
    """
    return [porter_stemmer.stem(word) for word in tokens]

def remove_stopwords(tokens: List[str], stopwords: set) -> List[str]:
    """
    Menghapus stopword (kata yang tidak bermakna penting).
    Input:
        tokens: list of words
        stopwords: set berisi kata-kata yang dihapus
    Output:
        filtered tokens
    """
    if not stopwords:
        return tokens
    return [t for t in tokens if t not in stopwords]


def document_preprocess(query_path : str, stopwords: set, use_stemming=False) -> List[str]:
    """
    Pipeline preprocessing lengkap untuk 1 dokumen.
    Output:
        list kata siap digunakan untuk BoW dan TF-IDF
    """
    with open(query_path, encoding="utf-8") as f:
        text = f.read()

    toks = tokenize(text)
    toks = normalize_tokens(toks)
    if use_stemming:
        toks = stem_tokens(toks)
    toks = remove_stopwords(toks, stopwords)
    return toks

# 2.2.2 MATRIKS TERM-DOCUMENT

def build_vocabulary(preprocessed_docs: List[List[str]]) -> List[str]:
    """
    Menghasilkan vocabulary unik (list term) dari seluruh dokumen.
    Urutan bebas tapi harus konsisten.
    Output:
        vocab: list[str] ukuran m
    """
    seen = {}
    vocab = []
    for doc in preprocessed_docs:
        for term in doc:
            if term not in seen:
                seen[term] = True
                vocab.append(term)
    return vocab


def build_term_document_matrix(preprocessed_docs: List[List[str]], vocab: List[str]) -> np.ndarray:
    """
    Membangun matriks term-document A ukuran (m x n)
    A[i][j] = frekuensi term ke-i pada dokumen ke-j

    Output:
        A: numpy array shape (m, n)
    """
    m = len(vocab)
    n = len(preprocessed_docs)
    tdm = np.zeros((m, n), dtype= np.float64)
    index = {term: idx for idx, term in enumerate(vocab)}
    for j, doc in enumerate(preprocessed_docs):
        for term in doc:
            if term in index:
                tdm[index[term], j] += 1.0
    return tdm

# 2.2.3 PEMBOBOTAN TF-IDF

def compute_tf(A: np.ndarray) -> np.ndarray:
    """
    Hitung TF berdasarkan rumus:
        TF_ij = A_ij / sum(A_*,j)

    TF shape harus sama dengan A.
    """
    col_sums = A.sum(axis=0)  
    tf = np.zeros_like(A, dtype=np.float64)
    nonzero_cols = col_sums > 0
    if np.any(nonzero_cols):
        tf[:, nonzero_cols] = A[:, nonzero_cols] / col_sums[nonzero_cols]
    return tf


def compute_idf(A: np.ndarray) -> np.ndarray:
    """
    Hitung IDF:
        IDF_i = log10( n / (1 + df_i) )

    Output:
        idf_vector ukuran (m,)
    """
    n_docs = A.shape[1]
    df = np.count_nonzero(A > 0, axis=1)  
    with np.errstate(divide='ignore'):
        idf = np.log10(n_docs / (1.0 + df))
    return idf


def compute_tfidf(A: np.ndarray) -> np.ndarray:
    """
    Menghitung TF-IDF:
        TFIDF = diag(IDF) x TF

    Output:
        TFIDF matrix shape (m, n)
    """
    tf = compute_tf(A) 
    idf = compute_idf(A)  
    tfidf = tf * idf[:, np.newaxis]
    return tfidf

# 2.2.4 TRUNCATED SVD (manual, via Eigen + Gram trick)

def _power_iteration_sym(mat: np.ndarray, num_iter: int = 1000, tol: float = 1e-6, rng_seed: Optional[int] = None) -> Tuple[float, np.ndarray]:
    """
    Melakukan power iteration pada matriks simetris `mat` untuk memperoleh eigenvalue terbesar beserta eigenvector-nya.
    - `mat` harus berupa matriks simetris.
    - Mengembalikan tuple (lambda, v) di mana:
        lambda : eigenvalue terbesar
        v      : eigenvector ter-normalisasi
    """
    n = mat.shape[0]
    if rng_seed is not None:
        random.seed(rng_seed)
    b = np.random.randn(n).astype(float)
    b = b / (math.sqrt((b * b).sum()) + 1e-16)
    lambda_old = 0.0
    for _ in range(num_iter):
        b_next = mat @ b
        norm_b_next = math.sqrt((b_next * b_next).sum())
        if norm_b_next == 0:
            return 0.0, np.zeros_like(b)
        b_next = b_next / norm_b_next
        lambda_new = float(b_next @ (mat @ b_next))
        if abs(lambda_new - lambda_old) < tol:
            return lambda_new, b_next
        b = b_next
        lambda_old = lambda_new
    lambda_final = float(b @ (mat @ b))
    return lambda_final, b

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
    m, n = A.shape
    use_ATA = (n <= m)
    if use_ATA:
        S = A.T @ A  
        dim = n
    else:
        S = A @ A.T  
        dim = m

    S = (S + S.T) / 2.0

    eigvals = []
    eigvecs = []

    S_work = S.copy()
    for i in range(k):
        lam, vec = _power_iteration_sym(S_work, num_iter=2000, tol=1e-6)
        if lam <= 0 or np.allclose(vec, 0):
            break
        eigvals.append(lam)
        eigvecs.append(vec.copy())
        S_work = S_work - lam * np.outer(vec, vec)
        S_work = (S_work + S_work.T) / 2.0

    k_found = len(eigvals)
    if k_found == 0:
        U_k = np.zeros((m, 0))
        Sigma_k = np.zeros((0, 0))
        V_k = np.zeros((n, 0))
        return U_k, Sigma_k, V_k

    eigvals = np.array(eigvals)
    eigvecs = np.column_stack(eigvecs)  

    order = np.argsort(-eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    sigma = np.sqrt(np.maximum(eigvals, 0.0))  # shape (k_found,)

    if use_ATA:
        V_k = eigvecs  
        U_list = []
        for i in range(k_found):
            v = V_k[:, i]
            if sigma[i] <= 1e-12:
                u = np.zeros((m,), dtype=float)
            else:
                u = (A @ v) / sigma[i]
                norm_u = math.sqrt((u * u).sum())
                if norm_u > 0:
                    u = u / norm_u
            U_list.append(u)
        U_k = np.column_stack(U_list)  
    else:
        U_k = eigvecs  
        V_list = []
        for i in range(k_found):
            u = U_k[:, i]
            if sigma[i] <= 1e-12:
                v = np.zeros((n,), dtype=float)
            else:
                v = (A.T @ u) / sigma[i]
                norm_v = math.sqrt((v * v).sum())
                if norm_v > 0:
                    v = v / norm_v
            V_list.append(v)
        V_k = np.column_stack(V_list)  

    Sigma_k = np.diag(sigma)

    if k_found < k:
        U_pad = np.zeros((m, k - k_found))
        V_pad = np.zeros((n, k - k_found))
        Sigma_pad = np.zeros((k - k_found, k - k_found))
        U_k = np.hstack([U_k, U_pad])
        V_k = np.hstack([V_k, V_pad])
        Sigma_k = np.block([
            [Sigma_k, np.zeros((k_found, k - k_found))],
            [np.zeros((k - k_found, k_found)), Sigma_pad]
        ])

    return U_k, Sigma_k, V_k

# 2.2.5 REPRESENTASI DOKUMEN DALAM RUANG SEMANTIK LATEN

def build_document_embeddings(V_k: np.ndarray, Sigma_k: np.ndarray) -> np.ndarray:
    """
    Embedding dokumen dihitung menggunakan:
        D = V_k x Σ_k

    Output:
        D shape (n x k)
        baris ke-i = embedding dokumen ke-i
    """
    if V_k.size == 0 or Sigma_k.size == 0:
        return np.zeros((V_k.shape[0], 0))
    return np.dot(V_k, Sigma_k)

# 2.2.6 METODE PERHITUNGAN SIMILARITAS

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Menormalisasi vector menjadi unit length.
    """
    norm = math.sqrt((v * v).sum())
    if norm <= 1e-12:
        return v
    return v / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Menghitung cosine similarity:
        sim(a, b) = (a ⋅ b) / (||a|| ||b||)

    Output:
        nilai float antara -1 hingga 1
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = math.sqrt((a * a).sum())
    nb = math.sqrt((b * b).sum())
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b)/ (na * nb))


def compute_similarity_scores(query_vec: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
    """
    Menghitung similarity query terhadap seluruh dokumen.
    Output:
        array shape (n,), similarity scores
    """
    if query_vec.ndim != 1:
        query_vec = query_vec.ravel()
    qn = math.sqrt((query_vec * query_vec).sum())
    if qn <= 1e-12:
        return np.zeros((doc_embeddings.shape[0],), dtype=float)
    dots = np.dot(doc_embeddings, query_vec)  # (n,)
    doc_norms = np.sqrt((doc_embeddings * doc_embeddings).sum(axis=1))  # (n,)
    denom = doc_norms * qn
    with np.errstate(divide='ignore', invalid='ignore'):
        sims = np.where(denom > 0, dots / denom, 0.0)
    return sims

# HIGH LEVEL: BUILD MODEL + QUERY
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
# LOADER MODEL
import pickle

def load_lsa_model(model_path: str) -> Dict:
    """
    Memuat model LSA dari file pickle.
    Digunakan saat aplikasi utama run agar tidak perlu build model ulang setiap request.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def preprocess_file(args):
    path, stop_words, use_stemming = args
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    toks = tokenize(text)
    toks = normalize_tokens(toks)
    if use_stemming:
        toks = stem_tokens(toks)
    toks = remove_stopwords(toks, stop_words)
    return toks


def build_lsa_model(dataset_dir: str, model_save_path: str, stop_words: set = set(stopwords.words('english')),
    k: int = 50, use_stemming=False, overwrite: bool = False) -> Dict:

    start_time_total = time.time()  # TIMER TOTAL

    if os.path.exists(model_save_path) and not overwrite:
        print("Model LSA sudah ada.")
        return load_lsa_model(model_save_path)

    print(f"Memulai Training LSA... (k={k})")

    # ================= LOAD FILES =================
    t_load = time.time()
    docs_path_list = []
    if os.path.exists(dataset_dir):
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.lower().endswith('.txt'):
                    docs_path_list.append(os.path.join(root, file))
    print(f"[TIMER] Load file selesai dalam {time.time() - t_load:.4f} detik")

    N = len(docs_path_list)
    if N == 0:
        raise ValueError(f"Dataset kosong! Tidak ada file .txt ditemukan di {dataset_dir}")

    print(f"Jumlah dokumen: {N}")

    # ================= PREPROCESSING =================
    t_pre = time.time()
    # preprocessed = []
    # # raw_texts = []
    # for p in docs_path_list:
    #     # with open(p, 'r', encoding='utf-8', errors='ignore') as f:
    #     #     txt = f.read()
    #     # raw_texts.append(txt)
    #     toks = document_preprocess(p, stop_words, use_stemming=use_stemming)
    #     preprocessed.append(toks)
    # Bungkus argumen agar bisa dipass ke multiprocessing
    task_args = [(p, stop_words, use_stemming) for p in docs_path_list]

    # Gunakan semua core CPU
    with mp.Pool(mp.cpu_count()) as pool:
        preprocessed = pool.map(preprocess_file, task_args)
    print(f"[TIMER] Preprocessing selesai dalam {time.time() - t_pre:.4f} detik")

    # ================= VOCABULARY =================
    t_vocab = time.time()
    vocab = build_vocabulary(preprocessed)
    print(f"[TIMER] Build vocabulary selesai dalam {time.time() - t_vocab:.4f} detik")

    # ================= TERM-DOCUMENT MATRIX =================
    t_matrix = time.time()
    A = build_term_document_matrix(preprocessed, vocab)
    print(f"[TIMER] Build term-document matrix selesai dalam {time.time() - t_matrix:.4f} detik")

    # ================= TF-IDF =================
    t_tfidf = time.time()
    tfidf = compute_tfidf(A)
    print(f"[TIMER] TF-IDF selesai dalam {time.time() - t_tfidf:.4f} detik")

    # ================= SVD =================
    t_svd = time.time()
    U_k, Sigma_k, V_k = truncated_svd(tfidf, k=k)
    print(f"[TIMER] SVD selesai dalam {time.time() - t_svd:.4f} detik")

    # ================= EMBEDDINGS =================
    t_embed = time.time()
    embeddings = build_document_embeddings(V_k, Sigma_k)
    print(f"[TIMER] Build embeddings selesai dalam {time.time() - t_embed:.4f} detik")

    # ================= IDF =================
    t_idf = time.time()
    idf = compute_idf(A)
    print(f"[TIMER] Compute IDF selesai dalam {time.time() - t_idf:.4f} detik")

    # ================= SAVE MODEL =================
    t_save = time.time()
    model = {
        'vocab': vocab,
        'A': A,
        'tfidf': tfidf,
        'U_k': U_k,
        'Sigma_k': Sigma_k,
        'V_k': V_k,
        'embeddings': embeddings,
        'preprocessed_docs': preprocessed,
        'idf': idf,
        'docs_paths': docs_path_list
    }
    save_model(model_save_path, model)
    print(f"[TIMER] Save model selesai dalam {time.time() - t_save:.4f} detik")

    # ================= TOTAL =================
    print(f"[TIMER] Total waktu training LSA: {time.time() - start_time_total:.4f} detik")

    return model


def embed_query(query_path: str, model: Dict, stopwords: set, use_stemming=False) -> np.ndarray:
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
    vocab = model['vocab']
    idf = model['idf']  
    U_k = model['U_k']  
    Sigma_k = model['Sigma_k']  

    toks = document_preprocess(query_path, stopwords, use_stemming=use_stemming)
    m = len(vocab)
    q_vec = np.zeros((m,), dtype=float)
    idx = {term: i for i, term in enumerate(vocab)}
    for t in toks:
        if t in idx:
            q_vec[idx[t]] += 1.0
    s = q_vec.sum()
    if s > 0:
        q_tf = q_vec / s
    else:
        q_tf = q_vec
    q_tfidf = q_tf * idf  
    if U_k.size == 0:
        return np.zeros((0,))
    proj = q_tfidf @ U_k  
    sigma_diag = np.diag(Sigma_k) if Sigma_k.size > 0 else np.array([])
    inv_sigma = np.zeros_like(sigma_diag)
    small = 1e-12
    for i, s_val in enumerate(sigma_diag):
        if s_val > small:
            inv_sigma[i] = 1.0 / s_val
        else:
            inv_sigma[i] = 0.0
    q_embed = proj * inv_sigma
    return q_embed

def get_top_k_recommendations(query_path: str, model: Dict, stopwords: set, k: int = 5) -> List[Dict]:
    """
    Mendapatkan top-k dokumen paling mirip berdasarkan cosine similarity.

    Return list berisi tuple:
    (index_dokumen, similarity)
    """
    q_embed = embed_query(query_path, model, stopwords)
    embeddings = model['embeddings']  
    if embeddings.size == 0 or q_embed.size == 0:
        return []
    sims = compute_similarity_scores(q_embed, embeddings)
    order = np.argsort(-sims)

    docs_paths = model['docs_paths']
    
    top_k = []
    for idx in order[:k]:
        top_k.append({
            "index": int(idx),
            "similarity": float(sims[idx]),
            "path": docs_paths[idx],
        })
    return top_k

# 2.2.7 FUNGSI QUERY UTAMA 

def query_lsa(query_path: str, model: Dict, stop_words: set = set(stopwords.words('english')), top_k: int = 5, use_stemming=False):
    """
    Fungsi high-level untuk melakukan query LSA.
    Frontend atau komponen lain cukup memanggil fungsi ini.

    Input:
        query_path   : path txt document
        model        : dictionary hasil build_lsa_model()
        stopwords    : set stopword
        top_k        : berapa banyak dokumen teratas yang dikembalikan
        use_stemming : apakah stemming dipakai pada query

    Output:
        list of (index_dokumen, similarity_score)
    """
    if not query_path or not query_path.strip():
        return []

    return get_top_k_recommendations(
        query_path=query_path,
        model=model,
        stopwords=stop_words,
        k=top_k
    )

if __name__ == "__main__":
    MODEL_PATH = "data/model_lsa.pkl"
    lsa_model = build_lsa_model(
        dataset_dir= "data/txt",
        model_save_path= MODEL_PATH,
        use_stemming= True,
        k = 50,)
    
    query_result = query_lsa("data/txt/1.txt", lsa_model, top_k=5,use_stemming=True)
    for res in query_result:
        print(res)