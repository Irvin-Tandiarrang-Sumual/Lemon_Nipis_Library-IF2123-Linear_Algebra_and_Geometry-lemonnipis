# Tugas Besar 2 IF2123 Aljabar Linier dan Geometri — 2025/2026  
## Kelompok LemonNipis

### *Anggota*
| Nama | NIM |
|------|------|
| Niko Samuel Simanjuntak | 13524029 |
| Irvin Tandiarrang Sumual | 13524030 |
| Kalyca Nathania B. Manullang | 13524071 |
# 🍋 LemonNipis Library

Aplikasi web untuk pencarian dan rekomendasi buku menggunakan teknik **Image Similarity (PCA)** dan **Text Similarity (LSA)**.

## Daftar Isi

- [Fitur](#-fitur)
- [Tech Stack](#-tech-stack)
- [Struktur Project](#-struktur-project)
- [Setup & Installation](#-setup--installation)
- [Cara Menjalankan](#-cara-menjalankan)
- [API Documentation](#-api-documentation)
- [Kontribusi](#-kontribusi)

---

## Fitur

### Pencarian
- **Pencarian Judul** - Cari buku berdasarkan nama judul
- **Pencarian Gambar** - Upload cover buku, sistem akan mencari kesamaan visual
- **Pencarian Dokumen** - Upload file txt, sistem akan mencari kesamaan konten

### Rekomendasi
- **Rekomendasi LSA** - Rekomendasi buku berdasarkan kesamaan konten teks

### Detail Buku
- Lihat cover dan judul buku
- Baca konten lengkap buku
- Dapatkan rekomendasi buku serupa

---

## Tech Stack

### Frontend
- **Framework**: Next.js 14 (React)
- **Styling**: Tailwind CSS
- **UI Components**: NextUI
- **HTTP Client**: Fetch API
- **TypeScript**: Type safety

### Backend
- **Framework**: FastAPI (Python)
- **Server**: Uvicorn
- **CORS**: FastAPI CORS Middleware
- **File Handling**: FastAPI UploadFile

### Temu Balik
- **Image Similarity**: PCA (Principal Component Analysis)
- **Text Similarity**: LSA (Latent Semantic Analysis)

### Data
- **Format**: JSON (mapper), TXT (dokumen), JPG (cover)
- **Storage**: Local filesystem

---

## Struktur Project

```
algeo2-lemonnipis/
├── data/                          # Data terpusat
│   ├── mapper.json                # Mapping buku (ID, judul, cover, txt)
│   ├── covers/                    # Cover images (JPG)
│   ├── txt/                       # Dokumen buku (TXT)
│   └── uploads/                   # Uploaded files (temporary)
│
├── src/
│   ├── backend/                   # FastAPI backend
│   │   ├── main.py               # Entry point, API routes
│   │   ├── pca_model.pkl         # Trained PCA model
│   │   ├── lsa_model.pkl         # Trained LSA model
│   │   ├── image/
│   │   │   ├── __init__.py
│   │   │   └── image_processing.py   # PCA image similarity
│   │   └── document/
│   │       ├── __init__.py
│   │       └── document_processing.py # LSA text similarity
│   │
│   └── frontend/                  # Next.js frontend
│       ├── app/
│       │   ├── page.tsx          # Home page
│       │   ├── layout.tsx        # Root layout
│       │   ├── book-collection/
│       │   │   ├── page.tsx      # Book list page
│       │   │   └── [id]/         # Dynamic book detail page
│       │   └── search-result/
│       │       └── page.tsx      # Search results page
│       ├── components/
│       │   ├── navbar.tsx        # Navigation bar
│       │   ├── search-input.tsx  # Search input component
│       │   ├── book-detail/
│       │   │   ├── content-view.tsx      # Book content display
│       │   │   ├── detail-wrapper.tsx    # Book detail wrapper
│       │   │   └── recommendation-view.tsx # Recommendations
│       │   └── icons.tsx         # SVG icons & logo
│       ├── config/
│       │   └── api.ts            # API configuration
│       ├── public/
│       │   └── LemonNipis.png    # Logo
│       └── styles/
│           └── globals.css       # Global styles
│
├── .gitignore
├── README.md                      # This file
└── package.json
```

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- Node.js 18+
- npm atau yarn

### Clone Repository

```bash
git clone https://github.com/lemonnipis/algeo2-lemonnipis.git
cd algeo2-lemonnipis
```

### Setup Backend

```bash
# Navigate ke backend
cd src/backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install fastapi uvicorn python-multipart nltk pillow scikit-learn numpy scipy

# Download NLTK data (one time only)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Setup Frontend

```bash
# Navigate ke frontend
cd src/frontend

# Install dependencies
npm install
# atau
yarn install

# Create .env.local
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
```

### Prepare Data

Pastikan folder `data/` di root level sudah ada dengan struktur:

```
data/
├── mapper.json           # JSON mapping
├── covers/              # Cover JPG files
├── txt/                 # Text TXT files
└── uploads/            # Auto-created by backend
```

**Format mapper.json:**

```json
{
  "38427": {
    "title": "The World as Will and Idea (Vol. 1 of 3)",
    "cover": "covers/38427.jpg",
    "txt": "txt/38427.txt"
  }
}
```

---

## Cara Menjalankan

### Jalankan Backend

```bash
cd src/backend

# Activate venv (jika belum)
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate     # Windows

# Run server
python main.py
```

Server akan berjalan di `http://localhost:8000`

**Output yang diharapkan:**
```
============================================================
PATH CONFIGURATION
============================================================
BASE_PATH: C:\ITB\Semester 3\AlGeo\algeo2-lemonnipis
DATA_DIR: C:\ITB\Semester 3\AlGeo\algeo2-lemonnipis\data - 
...
============================================================
Starting Server
============================================================
Server Ready!
============================================================
```

### Jalankan Frontend

**Terminal baru:**

```bash
cd src/frontend

# Development mode
npm run dev
# atau
yarn dev
```

Frontend akan berjalan di `http://localhost:3000`

### Buka di Browser

```
http://localhost:3000
```

---

## API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### **GET** `/api/books`
Dapatkan semua buku dengan pagination

**Query Parameters:**
- `skip`: int (default: 0)
- `limit`: int (default: 15)

**Response:**
```json
{
  "total": 1000,
  "results": [
    {
      "id": "38427",
      "title": "The World as Will and Idea",
      "cover": "covers/38427.jpg",
      "txt": "txt/38427.txt"
    }
  ]
}
```

---

#### **GET** `/api/search`
Cari buku berdasarkan judul

**Query Parameters:**
- `q`: string (required) - Query pencarian
- `skip`: int (default: 0)
- `limit`: int (default: 15)

**Response:**
```json
{
  "query": "harry",
  "total": 5,
  "results": [...]
}
```

---

#### **GET** `/api/books/{book_id}/content`
Dapatkan detail dan konten buku

**Response:**
```json
{
  "id": "38427",
  "title": "The World as Will and Idea",
  "cover": "covers/38427.jpg",
  "content": "Lorem ipsum dolor sit amet..."
}
```

---

#### **GET** `/api/books/{book_id}/recommendation`
Dapatkan rekomendasi buku berdasarkan LSA

**Response:**
```json
{
  "buku_yang_dicari_ditemukan": false,
  "recommendations": [
    {
      "id": "12345",
      "title": "Similar Book",
      "cover": "covers/12345.jpg",
      "similarity": 0.85
    }
  ]
}
```

---

#### **POST** `/api/search/image`
Cari buku berdasarkan upload gambar

**Request:**
- Form data dengan file: `file` (JPG/PNG)

**Response:**
```json
{
  "uploaded_file": "search_20250105_153022.jpg",
  "uploaded_url": "/data/uploads/search_20250105_153022.jpg",
  "total": 5,
  "query_results": [
    {
      "id": "38427",
      "title": "The World as Will and Idea",
      "cover": "covers/38427.jpg",
      "similarity": 0.92
    }
  ]
}
```

---

#### **POST** `/api/search/document`
Cari buku berdasarkan upload dokumen TXT

**Request:**
- Form data dengan file: `file` (TXT)

**Response:**
```json
{
  "total": 5,
  "query_results": [...]
}
```

---

#### **GET** `/health`
Health check

**Response:**
```json
{
  "status": "ok"
}
```

---

## Fitur Utama

### 1. Image Similarity Search (PCA)

**Cara kerja:**
1. Sistem membaca semua cover images dari `data/covers/`
2. Extract fitur visual menggunakan PCA
3. User upload gambar
4. Sistem bandingkan dengan database dan return top-5 hasil

**Konfigurasi:**
- Target image size: 200x300 pixels
- PCA components: 50
- Model file: `src/backend/pca_model.pkl`

### 2. Text Similarity Search (LSA)

**Cara kerja:**
1. Sistem membaca semua txt files dari `data/txt/`
2. Extract fitur semantic menggunakan LSA dengan stemming
3. User upload txt file
4. Sistem bandingkan dan return top-5 hasil

**Konfigurasi:**
- LSA components: 50
- Stemming: Enabled (Porter Stemmer)
- Model file: `src/backend/lsa_model.pkl`

### 3. Rekomendasi (LSA Based)

**Cara kerja:**
1. User membaca buku tertentu
2. Sistem ambil konten buku tersebut
3. Query dengan LSA model
4. Return rekomendasi buku dengan konten serupa

---

## Konfigurasi

### Backend Configuration (`src/backend/main.py`)

```python
# Paths
BASE_PATH = Path(__file__).parent.parent.parent  # Root project
DATA_DIR = BASE_PATH / "data"
MAPPER_PATH = DATA_DIR / "mapper.json"

# CORS
CORS Origins: ["http://localhost:3000", "http://localhost:3001"]

# PCA Model
target_width = 200
target_height = 300
k = 50

# LSA Model
k = 50
use_stemming = True
```

### Frontend Configuration (`src/frontend/config/api.ts`)

```typescript
export const API_BASE_URL = 
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
```

---

## Testing

### Backend Testing

```bash
cd src/backend

# Test health endpoint
curl http://localhost:8000/health

# Test get all books
curl http://localhost:8000/api/books

# Test search by title
curl "http://localhost:8000/api/search?q=harry"
```

### Frontend Testing

```bash
cd src/frontend

# Run tests
npm run test

# Build untuk production
npm run build

# Start production server
npm run start
```

---

## Catatan Penting

### Data Format

**mapper.json harus berisi:**
```json
{
  "ID": {
    "title": "Judul Buku",
    "cover": "covers/ID.jpg",
    "txt": "txt/ID.txt"
  }
}
```

**File naming:**
- Cover: `{ID}.jpg` (e.g., `38427.jpg`)
- Text: `{ID}.txt` (e.g., `38427.txt`)
- ID harus unique dan match di mapper

### Performance Tips

1. **Large datasets**: Update `limit` parameter di pagination
2. **Model training**: Models di-cache di `src/backend/*.pkl`
3. **Image size**: Standardize ke 200x300 untuk consistency

---

## roubleshooting

### Backend tidak start

```bash
# Check Python version
python --version  # harus 3.9+

# Check dependencies
pip list | grep fastapi

# Check port 8000 sudah digunakan?
# Windows:
netstat -ano | findstr :8000
# Linux/Mac:
lsof -i :8000

# Kill process dan restart
```

### NLTK data missing

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Models tidak ter-load

```
- Pastikan data/covers/ dan data/txt/ ada
- Pastikan file format JPG dan TXT valid
- Cek console untuk error messages
```

### Frontend API error

```bash
# Cek .env.local
cat src/frontend/.env.local

# Harus berisi:
# NEXT_PUBLIC_API_URL=http://localhost:8000

# Restart Next.js dev server
npm run dev
```

---

## Dependencies

### Backend (`requirements.txt`)

```
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
nltk==3.8.1
pillow==10.1.0
scikit-learn==1.3.2
numpy==1.26.2
scipy==1.11.4
```

### Frontend

```json
{
  "next": "14.0.0",
  "react": "^18.2.0",
  "tailwindcss": "^3.3.0",
  "@nextui-org/react": "^2.2.0"
}
```

---

## 📄 License

MIT License - Bebas digunakan untuk keperluan apapun

---

**Happy Searching! 🍋**
