# app/main.py
from fastapi import FastAPI, HTTPException
import os
import psycopg2
from psycopg2 import pool

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL non défini")

app = FastAPI(title="ML Model API", version="0.1.0")

# --- Pool de connexions ---
db_pool = pool.SimpleConnectionPool(minconn=1, maxconn=5, dsn=DATABASE_URL)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def read_root():
    return {"message": "FastAPI fonctionne 🎉"}

@app.get("/db-test")
def db_test():
    conn = None
    try:
        conn = db_pool.getconn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            result = cur.fetchone()
        return {"db": result[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")
    finally:
        if conn:
            db_pool.putconn(conn)

