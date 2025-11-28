from typing import List, Dict, Tuple
from backend.document.document_processing import embed_query, get_top_k_recommendations

class QueryService:
    """
    Wrapper untuk mempermudah pemanggilan fungsi similarity LSA.
    Frontend cukup memanggil:
        qs = QueryService(model, stopwords)
        result = qs.query("example text", top_k=5)
    """

    def __init__(self, model: Dict, stopwords: set, use_stemming: bool = False):
        """
        model: dictionary hasil dari build_lsa_model()
        stopwords: set of removed terms
        use_stemming: apakah stemming dipakai pada query
        """
        self.model = model
        self.stopwords = stopwords
        self.use_stemming = use_stemming

    def query(self, query_text: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Melakukan pencarian top-k dokumen paling mirip.

        Output format:
            [
                (index_dokumen, similarity_score),
                ...
            ]
        """
        if not query_text or not query_text.strip():
            return []

        results = get_top_k_recommendations(
            query_text=query_text,
            model=self.model,
            stopwords=self.stopwords,
            k=top_k
        )

        return results

    def query_with_raw_embedding(self, query_text: str):
        """
        Mengembalikan embedding query mentah juga jika frontend butuh untuk debugging atau visualisasi.
        """
        q_embed = embed_query(
            query_text=query_text,
            model=self.model,
            stopwords=self.stopwords,
            use_stemming=self.use_stemming
        )
        return q_embed
