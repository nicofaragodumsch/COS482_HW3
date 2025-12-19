import pymongo
from pymongo import MongoClient
import os
import csv

# ==========================================
# CONFIGURATION
# ==========================================
# Assuming files are in a subdirectory named 'IMDB' relative to this script
DATA_DIR = './IMDB' 
DB_NAME = 'imdb_assignment'
COLLECTION_NAME = 'movies'
CONNECTION_STRING = 'mongodb://localhost:27017/'

# Updated based on user input: using comma as delimiter
DELIMITER = ',' 

def get_file_path(filename):
    return os.path.join(DATA_DIR, filename)

def main():
    # 1. Connect to MongoDB
    try:
        client = MongoClient(CONNECTION_STRING)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Clear existing data to ensure a clean run for the assignment submission
        collection.drop()
        print(f"Connected to MongoDB. Collection '{COLLECTION_NAME}' cleared.")
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return

    # ==========================================
    # STEP 1: LOAD DATA INTO MEMORY
    # We load metadata into dictionaries first to facilitate embedding 
    # into a single document structure (Denormalization).
    # ==========================================
    
    # Data Structures
    movies = {}    # Key: mid, Value: Movie Dict
    persons = {}   # Key: pid, Value: Person Dict
    directors = {} # Key: did, Value: Director Dict

    print("Reading text files...")

    # A. Read Movies (IMDBMovie.txt)
    # Expected Columns: id, name, year, rank
    try:
        with open(get_file_path('IMDBMovie.txt'), 'r', encoding='utf-8', errors='replace') as f:
            # Using csv.reader handles potential edge cases with commas inside quotes
            reader = csv.reader(f, delimiter=DELIMITER)
            for parts in reader:
                if len(parts) >= 3: 
                    mid = parts[0].strip()
                    # Create the base document structure
                    movies[mid] = {
                        "_id": mid, # Use original ID as _id
                        "name": parts[1].strip(),
                        "year": parts[2].strip(),
                        "rank": parts[3].strip() if len(parts) > 3 else None,
                        "cast": [],      # Initialize embedded array for Cast
                        "directors": []  # Initialize embedded array for Directors
                    }
    except FileNotFoundError:
        print("Error: IMDBMovie.txt not found. Check your DATA_DIR.")
        return

    # B. Read Persons/Actors (IMDBPerson.txt)
    # Expected Columns: id, fname, lname, gender
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
        print("Error: IMDBPerson.txt not found.")

    # C. Read Directors Info (IMDBDirectors.txt)
    # Expected Columns: id, fname, lname
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
    # STEP 2: EMBED DATA (JOIN LOGIC)
    # ==========================================

    # D. Process Cast (IMDBCast.txt)
    # Expected Columns: pid, mid, role
    try:
        with open(get_file_path('IMDBCast.txt'), 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f, delimiter=DELIMITER)
            for parts in reader:
                if len(parts) >= 3:
                    pid, mid, role = parts[0].strip(), parts[1].strip(), parts[2].strip()
                    
                    # If we have valid references to Movie and Person, embed the data
                    if mid in movies and pid in persons:
                        actor_info = persons[pid].copy()
                        actor_info['role'] = role # Add the role specific to this movie
                        movies[mid]['cast'].append(actor_info)
    except FileNotFoundError:
        print("Error: IMDBCast.txt not found.")

    # E. Process Movie Directors (IMDBMovie_Directors.txt)
    # Expected Columns: did, mid
    try:
        with open(get_file_path('IMDBMovie_Directors.txt'), 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f, delimiter=DELIMITER)
            for parts in reader:
                if len(parts) >= 2:
                    did, mid = parts[0].strip(), parts[1].strip()
                    
                    # If we have valid references, embed the data
                    if mid in movies and did in directors:
                        director_info = directors[did].copy()
                        movies[mid]['directors'].append(director_info)
    except FileNotFoundError:
        print("Error: IMDBMovie_Directors.txt not found.")

    # ==========================================
    # STEP 3: INSERT INTO MONGODB (Task 1a)
    # ==========================================
    
    print(f"Preparing to insert {len(movies)} documents...")
    
    if movies:
        # Convert dictionary values to list for bulk insertion
        movie_documents = list(movies.values())
        # Use insert_many for performance
        collection.insert_many(movie_documents)
        print("Data insertion complete.")
    else:
        print("No movie data parsed.")

    # ==========================================
    # OPTIMIZATION (Task 1a - Design Shard Key/Index)
    # The assignment asks to design a key for fast access by 'name'.
    # In MongoDB, we implement this by creating an Index.
    # ==========================================
    print("Creating index on 'name' field for fast access...")
    collection.create_index("name")

    # ==========================================
    # STEP 4: QUERY (Task 1b)
    # Query database for 'Shrek (2001)'
    # ==========================================
    print("\n--- Task 1(b) Query Result ---")
    query_name = "Shrek (2001)"
    
    # Fetch the document
    result = collection.find_one({"name": query_name})
    
    if result:
        print(f"Movie Found: {result.get('name')}")
        print(f"Year: {result.get('year')}")
        print(f"Rank: {result.get('rank')}")
        
        # Display Directors
        director_names = [f"{d['fname']} {d['lname']}" for d in result.get('directors', [])]
        print(f"Directors: {', '.join(director_names)}")
        
        # Display Cast count and sample
        cast_list = result.get('cast', [])
        print(f"Total Cast Members: {len(cast_list)}")
        print("Cast (First 5):")
        for actor in cast_list[:5]:
            print(f" - {actor['fname']} {actor['lname']} as '{actor['role']}'")
    else:
        print(f"Movie '{query_name}' not found.")


if __name__ == "__main__":
    main()