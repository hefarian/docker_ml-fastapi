from fastapi import FastAPI
import psycopg2
import os

DATABASE_URL = os.getenv("DATABASE_URL")

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "FastAPI fonctionne 🎉"}

@app.get("/db-test")
def db_test():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("SELECT 1;")
    result = cur.fetchone()
    cur.close()
    conn.close()
    return {"db": result}
