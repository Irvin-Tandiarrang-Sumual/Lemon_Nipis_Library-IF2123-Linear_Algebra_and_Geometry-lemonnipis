from fastapi import FastAPI, UploadFile, Depends, Request, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
from image.image_processing import build_pca_model, query_image_from_model
from tempfile import NamedTemporaryFile
from document.document_processing import build_lsa_model, query_lsa
import shutil
import json
import os
# data path
BASE_PATH = Path(__file__).parent.parent.parent

# covers
COVERS_DIR_PATH = BASE_PATH / "data"

# txt
TXT_DIR_PATH = BASE_PATH / "data"

# mapper
MAPPER_PATH = BASE_PATH / "data" / "mapper.json"

# pca model
PCA_MODEL_PATH = Path(__file__).parent / "pca_model.pkl"

# lsa model
LSA_MODEL_PATH = Path(__file__).parent / "lsa_model.pkl"

def load_mapper(path : str):
    if not os.path.exists(path):
        print(f"Mapper file {path} tidak ditemukan")
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
@asynccontextmanager
async def lifespan(app:FastAPI):
    print("Starting Server")

    # load mapper
    print("Load mapper ...")
    app.state.book_mapper = load_mapper(str(MAPPER_PATH))
    print("Load mapper success ...")

    print("Building pca model ...")
    # load pca model
    app.state.pca_model = build_pca_model(
        dataset_dir= str(COVERS_DIR_PATH),
        model_save_path= str(PCA_MODEL_PATH),
        target_width=200,
        target_height=300,
        k=50,
    )
    print("Building pca model success")

    print("Building LSA model...")
    # load lsa model
    app.state.lsa_model = build_lsa_model(
        dataset_dir= str(TXT_DIR_PATH),
        model_save_path= str(LSA_MODEL_PATH),
        k = 50,
        use_stemming= True,
    )
    print("Building lsa model success ")

    yield
    print("Shutting Down Server")

app = FastAPI(lifespan=lifespan)

# dependency
async def get_book_mapper(request: Request):
    return request.app.state.book_mapper

async def get_pca_model(request: Request):
    return request.app.state.pca_model

async def get_lsa_model(request: Request):
    return request.app.state.lsa_model

app.add_middleware(
    CORSMiddleware,
    allow_origins = [
        "http://localhost:3000"
    ],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

@app.get("/")
async def read_root():
    return {"message" : "Hallo sar!"}

# (get) all book
@app.get("/api/books")
async def read_books(skip : int = 0, limit : int = 15,
                    book_mapper : dict = Depends(get_book_mapper)):
    if not book_mapper:
        raise HTTPException(status_code=503, detail="Mapper not loaded")
    all_books = []
    for book_id, book in book_mapper.items():
        all_books.append(
            {
                "id" : book_id,
                "title" : book.get("title", "Unknown"),
                "cover" : book.get("cover", ""),
                "txt" : book.get("txt", "")
            }
        )
    total = len(all_books)
    return {
        "total" : total,
        "paginated_results" : all_books[skip:skip + limit]
    }

# (get) detail - content
@app.get("/api/books/{book_id}/content")
async def read_book_detail_content(book_id : str, book_mapper : dict = Depends(get_book_mapper)):
    if not book_mapper:
        raise HTTPException(status_code=503, detail="Mapper not loaded")
    
    if book_id not in book_mapper:
        raise HTTPException(status_code=404, detail= "Book not found")
    
    print(f"Book id : {book_id}")
    book = book_mapper[book_id]

    # ambil path masing-masing
    cover_path = book.get("cover", "")
    txt_relative_path = book.get("txt", "")
    if not txt_relative_path:
        raise HTTPException(status_code=404, detail= "Text file path not found")
    txt_path = TXT_DIR_PATH / txt_relative_path
    if not txt_path.exists():
        raise HTTPException(status_code=404, detail= "Text file not found")
    
    try:
        with open(txt_path, 'r', encoding= 'utf-8') as f:
            content = f.read()
        return {
            "id" : book_id,
            "title" : book.get("title", "Unknown"),
            "cover" : cover_path,
            "content" : content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# (get) detail - recommendation
@app.get("/api/books/{book_id}/recommendation")
async def read_book_detail_recommendation(book_id : str,
            book_mapper : dict = Depends(get_book_mapper),
            lsa_model = Depends(get_lsa_model)):
    
    if not book_mapper:
        raise HTTPException(status_code=503, detail="Mapper not loaded")
    if book_id not in book_mapper:
        raise HTTPException(status_code=404, detail= "Book ID not found")
    
    top_k = 6
    book = book_mapper[book_id]
    doc_path = TXT_DIR_PATH / book.get("txt")
    query_results = query_lsa(str(doc_path), lsa_model, top_k= top_k)
    buku_yang_dicari_ditemukan = False
    delete_idx = top_k - 1
    for i, result in enumerate(query_results):
        filename = os.path.basename(result["path"])
        result_book_id = os.path.splitext(filename)[0]
        result_book = book_mapper[result_book_id]
        result["id"] = result_book_id
        result["title"] = result_book.get("title", "")
        result["cover"] = result_book.get("cover", "")
        if result_book_id == book_id:
            # sama dengan yg mo dicari yg mirpnya
            buku_yang_dicari_ditemukan = True
            delete_idx = i
    if buku_yang_dicari_ditemukan:
        del query_results[delete_idx]

    return {
        "buku_yang_dicari_ditemukan" : buku_yang_dicari_ditemukan,
        "recommendations" : query_results
    }

# (get) search pakai judul
@app.get("/api/books/search")
async def search_books_by_title(title_query : str, skip : int = 0, limit :int = 15, book_mapper : dict = Depends(get_book_mapper)):
    if not book_mapper:
        raise HTTPException(status_code=503, detail="Mapper not loaded")
    
    if not title_query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # proses cari pakai substring
    search_result = []
    for book_id, book in book_mapper.items():
        book_title = book.get("title", "Unknown")
        if (title_query.lower() in book_title.lower()): #biar case-insensitive
            search_result.append(
                {
                    "id" : book_id,
                    "title" : book_title,
                    "cover" : book.get("cover", ""),
                    "txt" : book.get("txt", "")
                }
            )

    # Pagination
    total = len(search_result)
    paginated_result = search_result[skip:skip + limit]
    return {
        "query" : title_query,
        "total" : total,
        "results" : paginated_result
    }


# (post) search pake image
@app.post("/api/books/search-by-image")
async def search_books_by_image(file : UploadFile = File(...),
                            book_mapper : dict = Depends(get_book_mapper),
                            pca_model = Depends(get_pca_model)):
    if not file:
        raise HTTPException(status_code=400, detail="File is empty")
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="File must be image")
    
    if not book_mapper:
        raise HTTPException(status_code=503, detail="Mapper not loaded")
    
    if not pca_model:
        raise HTTPException(status_code=503, detail="PCA Model not loaded")
    
    # siapin path untuk si file yg di-upload supaya bisa di-process
    with NamedTemporaryFile(delete=False) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    # proses cari results
    query_results = query_image_from_model (temp_file_path, pca_model, top_n= 5)

    for result in query_results:
        # cari id dari file name "ID.jpg"
        book_id = result.get("file_name").split('.')[0]
        book = book_mapper[book_id]
        result["id"] = book_id
        result["title"] = book.get("title", "")
        result["cover"] = book.get("cover", "")
    return {
        "uploaded_image_path" : temp_file_path,
        "query_results" : query_results
    }

