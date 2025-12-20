import subprocess
import pymongo
import sys
import os

# ==========================================
# CONFIGURATION
# ==========================================
STUDENT_SCRIPT = "task1_mongodb.py"
DB_NAME = "imdb_assignment"
COLLECTION_NAME = "movies"
REQUIRED_INDEX = "name"

def run_student_script():
    """Runs the student script and captures stdout/stderr."""
    print(f"Running {STUDENT_SCRIPT}...")
    print("⏳ This may take a minute or two depending on your computer's speed...")
    try:
        # TIMEOUT SET TO 5 MINUTES
        result = subprocess.run(
            [sys.executable, STUDENT_SCRIPT],
            capture_output=True,
            text=True,
            timeout=300  
        )
        return result
    except subprocess.TimeoutExpired:
        print("❌ Error: Script timed out after 5 minutes.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Could not find {STUDENT_SCRIPT}.")
        sys.exit(1)

def validate_checklist(script_output):
    """Validates checklist items against Output and DB State."""
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    report = []
    
    # ---------------------------------------------------------
    # CHECK 1: Script Execution
    # ---------------------------------------------------------
    if script_output.returncode == 0:
        report.append(("[x] Script Execution", "PASS", "Script ran without errors."))
    else:
        print("\n--- SCRIPT ERROR OUTPUT ---")
        print(script_output.stderr)
        report.append(("[ ] Script Execution", "FAIL", "Script failed."))
        return report 

    # ---------------------------------------------------------
    # CHECK 2: Database Selection
    # ---------------------------------------------------------
    if DB_NAME in client.list_database_names():
        report.append(("[x] Database Selection", "PASS", f"Database '{DB_NAME}' created."))
    else:
        report.append(("[ ] Database Selection", "FAIL", f"Database '{DB_NAME}' not found."))
        return report

    db = client[DB_NAME]
    
    # ---------------------------------------------------------
    # CHECK 3: Collection Creation
    # ---------------------------------------------------------
    if COLLECTION_NAME in db.list_collection_names():
        report.append(("[x] Collection Creation", "PASS", f"Collection '{COLLECTION_NAME}' exists."))
    else:
        report.append(("[ ] Collection Creation", "FAIL", f"Collection '{COLLECTION_NAME}' not found."))
        return report

    collection = db[COLLECTION_NAME]

    # ---------------------------------------------------------
    # CHECK 4: Data Storage Model (Single Collection + Embedding)
    # ---------------------------------------------------------
    # IMPROVED CHECK: Find a movie that definitely has cast members (like Shrek)
    # instead of a random one that might be empty.
    sample_doc = collection.find_one({"name": "Shrek (2001)"})

    if sample_doc:
        # Check if 'cast' and 'directors' keys exist and are lists
        is_cast_list = isinstance(sample_doc.get('cast'), list)
        is_dir_list = isinstance(sample_doc.get('directors'), list)
        
        # Check if they actually contain data (since Shrek should)
        has_cast_data = len(sample_doc.get('cast', [])) > 0
        
        if is_cast_list and is_dir_list and has_cast_data:
             report.append(("[x] Data Storage Model", "PASS", "Document contains correctly embedded 'cast' and 'directors' arrays."))
        else:
             report.append(("[ ] Data Storage Model", "FAIL", "Embedded arrays are missing or empty in 'Shrek (2001)'."))
    else:
        report.append(("[ ] Data Storage Model", "FAIL", "Could not find 'Shrek (2001)' to verify model."))

    # ---------------------------------------------------------
    # CHECK 5: File Parsing (Data Volume)
    # ---------------------------------------------------------
    count = collection.count_documents({})
    if count > 0:
        report.append(("[x] File Parsing", "PASS", f"Successfully imported {count} documents."))
    else:
        report.append(("[ ] File Parsing", "FAIL", "Collection is empty."))

    # ---------------------------------------------------------
    # CHECK 6: Optimization (Index on 'name')
    # ---------------------------------------------------------
    indexes = collection.index_information()
    index_found = False
    for name, details in indexes.items():
        if details['key'][0][0] == REQUIRED_INDEX:
            index_found = True
            break
    
    if index_found:
        report.append(("[x] Optimization", "PASS", f"Index on '{REQUIRED_INDEX}' exists."))
    else:
        report.append(("[ ] Optimization", "FAIL", f"Index on '{REQUIRED_INDEX}' NOT found."))

    # ---------------------------------------------------------
    # CHECK 7: Task (b) Output Validation (Query Result)
    # ---------------------------------------------------------
    output_str = script_output.stdout
    if "Found Movie: Shrek (2001)" in output_str and "Directors: Andrew Adamson" in output_str:
        report.append(("[x] Task (b) Query Output", "PASS", "Script printed correct details for 'Shrek (2001)'."))
    else:
        report.append(("[ ] Task (b) Query Output", "FAIL", "Script output did not match expected query format."))

    return report

def main():
    result = run_student_script()
    results = validate_checklist(result)
    
    print("\n" + "="*40)
    print("      ASSIGNMENT CHECKLIST REPORT")
    print("="*40)
    
    all_passed = True
    for item, status, msg in results:
        print(f"{item:<25} | {status:<4} | {msg}")
        if status == "FAIL":
            all_passed = False
            
    print("-" * 80)
    if all_passed:
        print("✅ SUCCESS: All code requirements met!")
    else:
        print("❌ FAILURE: Some requirements were not met.")

if __name__ == "__main__":
    main()