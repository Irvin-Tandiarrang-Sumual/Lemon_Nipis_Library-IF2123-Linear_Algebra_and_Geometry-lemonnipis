from fastapi import FastAPI, UploadFile, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
from image.image_processing import build_pca_model
import json
import os
# data path
BASE_PATH = Path(__file__).parent.parent.parent

# covers
COVERS_DIR_PATH = BASE_PATH / "data" / "covers"

# txt
TXT_DIR_PATH = BASE_PATH / "data" / "txt"

# mapper
MAPPER_PATH = BASE_PATH / "data" / "mapper.json"

# pca model
PCA_MODEL_PATH = Path(__file__).parent / "pca_model.pkl"

def load_mapper(path : str):
    if not os.path.exists(path):
        print(f"Mapper file {path} tidak ditemukan")
        return {}
    with open(path, 'r') as f:
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


    # load lsa model (soon)
    yield
    print("Shutting Down Server")

app = FastAPI(lifespan=lifespan)

# dependency
async def get_book_mapper(request: Request):
    return request.app.state.book_mapper

async def get_pca_model(request: Request):
    return request.app.state.pca_model

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
async def read_books(book_mapper : dict = Depends(get_book_mapper)):
    return book_mapper

# (get) detail - content
@app.get("/api/books/{book_id}/content")
async def read_book_detail_content(book_id : str, book_mapper : dict = Depends(get_book_mapper)):
    pass

# (get) detail - recommendation
@app.get("/api/books/{book_id}/recommendation")
async def read_book_detail_recommendation(book_id : str, book_mapper : dict = Depends(get_book_mapper)):
    pass

# (get) search pakai judul
@app.get("/api/books/search")
async def search_books_by_title(title : str, book_mapper : dict = Depends(get_book_mapper)):
    pass

# (post) search pake image
@app.post("/api/books/search-by-image")
async def search_books_by_image(file : UploadFile):
    pass
