import os
import pickle
from typing import List, Set, Dict

from backend.document.document_processing import build_lsa_model


def load_documents_from_folder(folder_path: str) -> List[str]:
    """
    Mengambil seluruh path file dokumen (.txt) dari folder dataset.
    Urutan file akan menentukan index dokumennya.
    """
    docs = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.lower().endswith(".txt"):
            docs.append(os.path.join(folder_path, fname))
    return docs


def load_stopwords(path: str) -> Set[str]:
    """
    Memuat stopwords dari file.
    Jika file tidak ada, return set kosong.
    """
    if not os.path.exists(path):
        return set()

    stopwords = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip().lower()
            if s:
                stopwords.add(s)
    return stopwords


def save_model_pickle(model: Dict, output_path: str):
    """
    Menyimpan model ke file pickle.
    """
    with open(output_path, "wb") as f:
        pickle.dump(model, f)


def main():
    DOCUMENT_DIR = "data/documents"      
    STOPWORD_PATH = "data/stopwords.txt" 
    OUTPUT_MODEL = "model/model.pkl"    

    print("Loading documents...")
    doc_paths = load_documents_from_folder(DOCUMENT_DIR)
    print(f"  -> {len(doc_paths)} documents found.")

    print("Loading stopwords...")
    stopwords = load_stopwords(STOPWORD_PATH)
    print(f"  -> {len(stopwords)} stopwords loaded.")

    print("Building LSA model (this may take a while)...")
    model = build_lsa_model(
        docs_path_list=doc_paths,
        stopwords=stopwords,
        k=50,               
        use_stemming=False  
    )

    print("Saving model to pickle...")
    os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
    save_model_pickle(model, OUTPUT_MODEL)

    print("DONE. Model saved at:", OUTPUT_MODEL)


if __name__ == "__main__":
    main()
