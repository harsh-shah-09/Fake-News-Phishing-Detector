import os
import sqlite3
import urllib.request
import zipfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, '../backend/domains.db')
TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"
ZIP_PATH = os.path.join(BASE_DIR, "top-1m.csv.zip")
CSV_PATH = os.path.join(BASE_DIR, "top-1m.csv")

print("[INFO] Downloading Tranco Top 1M database...")
urllib.request.urlretrieve(TRANCO_URL, ZIP_PATH)

print("[INFO] Extracting CSV archive...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(BASE_DIR)

print("[INFO] Building indexed SQLite database...")
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("CREATE TABLE domains (domain TEXT PRIMARY KEY);")

batch = []
with open(CSV_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) == 2:
            batch.append((parts[1].lower(),))
            if len(batch) >= 50000:
                cursor.executemany("INSERT OR IGNORE INTO domains VALUES (?);", batch)
                batch = []

if batch:
    cursor.executemany("INSERT OR IGNORE INTO domains VALUES (?);", batch)

conn.commit()
conn.close()

if os.path.exists(ZIP_PATH):
    os.remove(ZIP_PATH)
if os.path.exists(CSV_PATH):
    os.remove(CSV_PATH)

print(f"[SUCCESS] Indexed SQLite database created at: {DB_PATH}")