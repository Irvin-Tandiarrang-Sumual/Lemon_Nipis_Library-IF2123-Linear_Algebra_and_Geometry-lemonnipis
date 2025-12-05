from fastapi import FastAPI, UploadFile, Depends, Request, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
from image.image_processing import build_pca_model, query_image_from_model
from document.document_processing import build_lsa_model, query_lsa
import shutil
import json
import os
from datetime import datetime
from fastapi.staticfiles import StaticFiles
# data path
BASE_PATH = Path(__file__).parent.parent.parent

# Folder data
DATA_DIR_PATH = BASE_PATH / "data"
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

UPLOADS_DIR_IMG = BASE_PATH / "data" / "uploads" / "image"
UPLOADS_DIR_TXT = BASE_PATH / "data" / "uploads" / "txt"
UPLOADS_DIR_IMG.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR_TXT.mkdir(parents=True, exist_ok=True)

def load_mapper(path : str):
    if not os.path.exists(path):
        print(f"Mapper file {path} tidak ditemukan")
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
def get_full_cover_url(request: Request, relative_path: str):
    if not relative_path:
        return ""
    clean_path = relative_path.replace("\\", "/")
    return f"{request.base_url}static/{clean_path}"

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
        dataset_dir= str(COVERS_DIR_PATH / "covers"),
        model_save_path= str(PCA_MODEL_PATH),
        target_width=200,
        target_height=300,
        k=50,
    )
    print("Building pca model success")

    print("Building LSA model...")
    # load lsa model
    app.state.lsa_model = build_lsa_model(
        dataset_dir= str(TXT_DIR_PATH / "txt"),
        model_save_path= str(LSA_MODEL_PATH),
        k = 50,
        use_stemming= True,
    )
    print("Building lsa model success ")

    yield
    print("Shutting Down Server")

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(DATA_DIR_PATH)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins = [
        "http://localhost:3000"
    ],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

async def get_book_mapper(request: Request):
    return request.app.state.book_mapper

async def get_pca_model(request: Request):
    return request.app.state.pca_model

async def get_lsa_model(request: Request):
    return request.app.state.lsa_model

@app.get("/")
async def read_root():
    return {"message" : "Hallo sar!"}

# (get) all book
@app.get("/api/books")
async def read_books(request: Request, skip : int = 0, limit : int = 15,
                    book_mapper : dict = Depends(get_book_mapper)):
    if not book_mapper:
        raise HTTPException(status_code=503, detail="Mapper not loaded")
    all_books = []
    for book_id, book in book_mapper.items():
        full_cover_url = get_full_cover_url(request, book.get("cover", ""))
        all_books.append(
            {
                "id" : book_id,
                "title" : book.get("title", "Unknown"),
                "cover" : full_cover_url,
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
async def read_book_detail_content(book_id : str, request : Request, book_mapper : dict = Depends(get_book_mapper)):
    if not book_mapper:
        raise HTTPException(status_code=503, detail="Mapper not loaded")
    
    if book_id not in book_mapper:
        raise HTTPException(status_code=404, detail= "Book not found")
    
    print(f"Book id : {book_id}")
    book = book_mapper[book_id]

    full_cover_url = get_full_cover_url(request, book.get("cover", ""))
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
            "cover" : full_cover_url,
            "content" : content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# (get) detail - recommendation
@app.get("/api/books/{book_id}/recommendation")
async def read_book_detail_recommendation(book_id : str, request : Request,
            book_mapper : dict = Depends(get_book_mapper),
            lsa_model = Depends(get_lsa_model)):
    
    if not book_mapper:
        raise HTTPException(status_code=503, detail="Mapper not loaded")
    if book_id not in book_mapper:
        raise HTTPException(status_code=404, detail= "Book ID not found")
    
    top_k = 6
    book = book_mapper[book_id]
    doc_path = TXT_DIR_PATH / book.get("txt")
    query_results = query_lsa(str(doc_path), lsa_model, top_k= top_k, use_stemming= True)
    buku_yang_dicari_ditemukan = False
    delete_idx = top_k - 1
    for i, result in enumerate(query_results):
        filename = os.path.basename(result["path"])
        result_book_id = os.path.splitext(filename)[0]
        result_book = book_mapper[result_book_id]
        full_cover_url = get_full_cover_url(request, result_book.get("cover", ""))
        result["id"] = result_book_id
        result["title"] = result_book.get("title", "")
        result["cover"] = full_cover_url
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
async def search_books_by_title(title_query : str, request : Request, skip : int = 0, limit :int = 15, book_mapper : dict = Depends(get_book_mapper)):
    if not book_mapper:
        raise HTTPException(status_code=503, detail="Mapper not loaded")
    
    if not title_query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # proses cari pakai substring
    search_result = []
    for book_id, book in book_mapper.items():
        book_title = book.get("title", "Unknown")
        if (title_query.lower() in book_title.lower()): #biar case-insensitive
            full_cover_url = get_full_cover_url(request, book.get("cover", ""))
            search_result.append(
                {
                    "id" : book_id,
                    "title" : book_title,
                    "cover" : full_cover_url,
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
async def search_books_by_image(request : Request, file : UploadFile = File(...),
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
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(file.filename).suffix
        saved_filename = f"search_{timestamp}{file_extension}"
        saved_path = UPLOADS_DIR_IMG / saved_filename

        with open(saved_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        query_results = query_image_from_model(str(saved_path), pca_model, top_n=5)
        
        formatted_results = []
        for result in query_results:
            # Asumsi result["file_name"] adalah "38427.jpg"
            book_id = result.get("file_name").split('.')[0]
            
            if book_id in book_mapper and result.get("score") <= 6000:
                print(f"SCOREE: {result.get("score")}")
                book = book_mapper[book_id]
                full_cover_url = get_full_cover_url(request, book.get("cover", ""))
                
                result["id"] = book_id
                result["title"] = book.get("title", "")
                result["cover"] = full_cover_url
                formatted_results.append(result)
        
        uploaded_image_url = f"{request.base_url}static/uploads/image/{saved_filename}"

        return {
            "uploaded_image_path": uploaded_image_url,
            "query_results": formatted_results
        }
    except Exception as e:
        print(f" Error : {e}")
        raise HTTPException(status_code=400, detail=str(e))

# (post) search pake document
@app.post("/api/books/search-by-document")
async def search_books_by_document(request : Request, file : UploadFile = File(...),
                            book_mapper : dict = Depends(get_book_mapper),
                            lsa_model = Depends(get_lsa_model)):
    if not file:
        raise HTTPException(status_code=400, detail="File is empty")
    
    if not file.filename.lower().endswith(('.txt')):
        raise HTTPException(status_code=400, detail="File must be txt")
    
    if not book_mapper:
        raise HTTPException(status_code=503, detail="Mapper not loaded")
    
    if not lsa_model:
        raise HTTPException(status_code=503, detail="LSA Model not loaded")

    # Simpan temp file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_filename = f"search_{timestamp}.txt"
    saved_path = UPLOADS_DIR_TXT / saved_filename
    
    with open(saved_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)

    # Proses query
    query_results = query_lsa(str(saved_path), lsa_model, top_k=5, use_stemming=True)

    formatted_results = []
    for i, result in enumerate(query_results):
        filename = os.path.basename(result["path"])
        result_book_id = os.path.splitext(filename)[0]
        
        if result_book_id in book_mapper:
            result_book = book_mapper[result_book_id]
            full_cover_url = get_full_cover_url(request, result_book.get("cover", ""))
            
            result["id"] = result_book_id
            result["title"] = result_book.get("title", "")
            result["cover"] = full_cover_url
            formatted_results.append(result)
    
    return {
        "query_results": formatted_results
    }