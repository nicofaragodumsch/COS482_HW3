import pymongo
from pymongo import MongoClient
import os
import csv
import sys

# ==========================================
# CONFIGURATION
# ==========================================
DB_NAME = 'imdb_assignment'
COLLECTION_NAME = 'movies'
CONNECTION_STRING = 'mongodb://localhost:27017/'
DELIMITER = ',' 

def get_data_directory():
    """
    Smartly determines where the data files are.
    Checks './IMDB' first, then the current directory '.'.
    """
    # Check if 'IMDB' folder exists and has the main file
    if os.path.exists('./IMDB') and os.path.exists(os.path.join('./IMDB', 'IMDBMovie.txt')):
        return './IMDB'
    # Check if files are in the current directory
    elif os.path.exists('IMDBMovie.txt'):
        return '.'
    else:
        return None

def main():
    # 1. Locate Data
    data_dir = get_data_directory()
    if data_dir is None:
        print("❌ CRITICAL ERROR: Could not find dataset files.")
        print("   Please ensure 'IMDBMovie.txt' is in the current folder OR inside an 'IMDB' subfolder.")
        sys.exit(1) # Force exit with error so the test script knows it failed
        
    print(f"📂 Found data files in: '{data_dir}'")
    
    def get_file_path(filename):
        return os.path.join(data_dir, filename)

    # 2. Connect to MongoDB
    try:
        client = MongoClient(CONNECTION_STRING)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Clear existing data
        collection.drop()
        print(f"Connected to MongoDB. Collection '{COLLECTION_NAME}' cleared.")
    except Exception as e:
        print(f"❌ Error connecting to MongoDB: {e}")
        sys.exit(1)

    # ==========================================
    # STEP 1: LOAD DATA INTO MEMORY
    # ==========================================
    
    movies = {}    
    persons = {}   
    directors = {} 

    print("Reading text files...")

    # A. Read Movies
    try:
        with open(get_file_path('IMDBMovie.txt'), 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f, delimiter=DELIMITER)
            for parts in reader:
                if len(parts) >= 3: 
                    mid = parts[0].strip()
                    movies[mid] = {
                        "_id": mid, 
                        "name": parts[1].strip(),
                        "year": parts[2].strip(),
                        "rank": parts[3].strip() if len(parts) > 3 else None,
                        "cast": [],      
                        "directors": [] 
                    }
    except FileNotFoundError:
        print("❌ Error: IMDBMovie.txt not found.")
        sys.exit(1)

    # B. Read Persons
    try:
        with open(get_file_path('IMDBPerson.txt'), 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f, delimiter=DELIMITER)
            for parts in reader:
                if len(parts) >= 4:
                    pid = parts[0].strip()
                    persons[pid] = {
                        "pid": pid,
                        "fname": parts[1].strip(),
                        "lname": parts[2].strip(),
                        "gender": parts[3].strip()
                    }
    except FileNotFoundError:
        print("Error: IMDBPerson.txt not found.") # Non-critical if missing, but ideally should exist

    # C. Read Directors
    try:
        with open(get_file_path('IMDBDirectors.txt'), 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f, delimiter=DELIMITER)
            for parts in reader:
                if len(parts) >= 3:
                    did = parts[0].strip()
                    directors[did] = {
                        "did": did,
                        "fname": parts[1].strip(),
                        "lname": parts[2].strip()
                    }
    except FileNotFoundError:
        print("Error: IMDBDirectors.txt not found.")

    # ==========================================
    # STEP 2: EMBED DATA
    # ==========================================

    # D. Process Cast
    try:
        with open(get_file_path('IMDBCast.txt'), 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f, delimiter=DELIMITER)
            for parts in reader:
                if len(parts) >= 3:
                    pid, mid, role = parts[0].strip(), parts[1].strip(), parts[2].strip()
                    if mid in movies and pid in persons:
                        actor_info = persons[pid].copy()
                        actor_info['role'] = role 
                        movies[mid]['cast'].append(actor_info)
    except FileNotFoundError:
        print("Warning: IMDBCast.txt not found (skipping cast embedding).")

    # E. Process Movie Directors
    try:
        with open(get_file_path('IMDBMovie_Directors.txt'), 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f, delimiter=DELIMITER)
            for parts in reader:
                if len(parts) >= 2:
                    did, mid = parts[0].strip(), parts[1].strip()
                    if mid in movies and did in directors:
                        director_info = directors[did].copy()
                        movies[mid]['directors'].append(director_info)
    except FileNotFoundError:
        print("Warning: IMDBMovie_Directors.txt not found (skipping director embedding).")

    # ==========================================
    # STEP 3: INSERT INTO MONGODB
    # ==========================================
    
    print(f"Preparing to insert {len(movies)} documents...")
    
    if movies:
        movie_documents = list(movies.values())
        collection.insert_many(movie_documents)
        print("✅ Data insertion complete.")
    else:
        print("❌ Error: No movie data parsed. Database will not be created.")
        sys.exit(1)

    # ==========================================
    # OPTIMIZATION (Task 1a)
    # ==========================================
    print("Creating index on 'name' field...")
    collection.create_index("name")

    # ==========================================
    # STEP 4: QUERY (Task 1b)
    # ==========================================
    print("\n--- Task 1(b) Query Result ---")
    query_name = "Shrek (2001)"
    result = collection.find_one({"name": query_name})
    
    if result:
        print(f"Found Movie: {result.get('name')}")
        print(f"Year: {result.get('year')}")
        print(f"Rank: {result.get('rank')}")
        director_names = [f"{d['fname']} {d['lname']}" for d in result.get('directors', [])]
        print(f"Directors: {', '.join(director_names)}")
        print(f"Cast Count: {len(result.get('cast', []))}")
    else:
        print(f"Movie '{query_name}' not found.")


if __name__ == "__main__":
    main()