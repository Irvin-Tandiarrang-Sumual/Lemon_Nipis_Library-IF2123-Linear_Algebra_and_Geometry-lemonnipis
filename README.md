[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/WQhgEYKW)
# Template Tugas Besar 2 Aljabar Linier dan Geometri

## How to Develop
The template base file structure is as below
```
root
├───data
├───docs
├───src
│   ├───backend
│   └───frontend
├───test
├───.gitignore
├───README.md
└───LICENSE
```

### Purpose of each directory
1. `data`: to store application data, this should include huge datasets that is used in the applicaiton
2. `docs`: to store final report and other documents
3. `src`: to store source code of the application. Includes `frontend` and `backend` to store each respective component of the app
4. `test`: to store test cases


### Run
Backend (src/backend): 
    1. Create : python -m venv venv
    2. Aktifkan Python Virtual Environment : venv\Scripts\activate
    3. Jalankan FastAPI-nya : uvicorn main:app --reload
Frontend (src/frontend) : 
    1. Jalankan FE : npm run dev