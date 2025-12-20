import unittest
import os
import shutil
import subprocess
import csv
import glob
import sys

# ==========================================
# CONFIGURATION
# ==========================================
STUDENT_SCRIPT = "task3_sparksql.py"
TEST_PR_FILE = "pagerank_output.txt"
TEST_NAMES_FILE = "names.txt"
OUTPUT_DIR = "task3_output_csv"

# Controlled test data
TEST_PR_DATA = [
    "0 0.10",
    "1 0.20",
    "2 0.30",  # Vertex 2 has rank 0.30
    "3 0.40"   # Max rank is 0.40 (Vertex 3)
]

TEST_NAMES_DATA = [
    "0 Adam",
    "1 Lisa",
    "2 Bert",
    "3 Ralph"
]

class TestTask3(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Prepare the environment:
        1. create controlled input files.
        2. Run the student script once.
        3. Capture stdout/stderr for analysis.
        """
        print(f"🔵 [SETUP] Creating test files '{TEST_PR_FILE}' and '{TEST_NAMES_FILE}'...")
        
        # 1. Create PageRank Input
        with open(TEST_PR_FILE, "w") as f:
            f.write("\n".join(TEST_PR_DATA) + "\n")

        # 2. Create Names Input
        with open(TEST_NAMES_FILE, "w") as f:
            f.write("\n".join(TEST_NAMES_DATA) + "\n")

        # 3. Remove existing output directory to ensure a fresh run
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)

        print(f"🔵 [RUNNING] Executing {STUDENT_SCRIPT}...")
        
        # 4. Run the student script
        try:
            cls.process = subprocess.run(
                [sys.executable, STUDENT_SCRIPT],
                capture_output=True,
                text=True,
                check=True
            )
            cls.stdout = cls.process.stdout
            cls.stderr = cls.process.stderr
            cls.exit_code = cls.process.returncode
        except subprocess.CalledProcessError as e:
            cls.process = None
            cls.stdout = e.stdout
            cls.stderr = e.stderr
            cls.exit_code = e.returncode
            print(f"❌ Script crashed with code {e.returncode}")
            print(e.stderr)

    def test_01_script_runs_successfully(self):
        """Check if script exits with code 0."""
        self.assertEqual(self.exit_code, 0, f"Script failed to run. Error: {self.stderr}")
        print("✅ [PASS] Script ran successfully (Exit Code 0).")

    def test_02_task_b_specific_vertex_query(self):
        """
        Task 3(b): Verify it queries for Vertex 2.
        Based on our test data, Vertex 2 has rank 0.30.
        Spark 'show()' output should contain '0.3'.
        """
        # We look for the float value in the output
        found = "0.3" in self.stdout
        self.assertTrue(found, "Could not find expected PageRank '0.3' for Vertex 2 in stdout.")
        print("✅ [PASS] Task 3(b) output found PageRank for Vertex 2.")

    def test_03_task_c_max_rank_query(self):
        """
        Task 3(c): Verify it finds the max rank vertex.
        Based on test data, max is Vertex 3 with 0.40.
        """
        # Look for the ID '3' and Rank '0.4' appearing in the output
        found_id = "3" in self.stdout
        found_rank = "0.4" in self.stdout
        
        self.assertTrue(found_id and found_rank, 
                        f"Could not find Max Vertex ID '3' or Rank '0.4' in stdout.\nStdout snippet: {self.stdout[-500:]}")
        print("✅ [PASS] Task 3(c) found Vertex with largest PageRank.")

    def test_04_task_e_csv_generated(self):
        """Task 3(e): Check if CSV output directory and file exist."""
        self.assertTrue(os.path.exists(OUTPUT_DIR), "Output directory 'task3_output_csv' was not created.")
        
        # Spark writes part-00000-....csv files inside the directory
        csv_files = glob.glob(os.path.join(OUTPUT_DIR, "*.csv"))
        self.assertTrue(len(csv_files) > 0, "No .csv file found inside the output directory.")
        
        self.csv_file_path = csv_files[0]
        print(f"✅ [PASS] CSV Output generated at {self.csv_file_path}.")

    def test_05_task_e_csv_content_join(self):
        """
        Task 3(e): Verify the content of the CSV confirms the JOIN worked.
        We expect 'Bert' (from names) and '0.3' (from PR) to be in the same row.
        """
        # Identify the CSV file (found in previous test)
        csv_files = glob.glob(os.path.join(OUTPUT_DIR, "*.csv"))
        if not csv_files:
            self.fail("Cannot test CSV content; file missing.")
            
        target_file = csv_files[0]
        
        found_join_entry = False
        with open(target_file, 'r') as f:
            content = f.read()
            # Simple string check for the joined data row: 2, Bert, 0.3
            # Spark CSVs might quote strings, e.g., "Bert", so we check loosely
            if "Bert" in content and "0.3" in content:
                found_join_entry = True
                
        self.assertTrue(found_join_entry, "CSV did not contain joined data (expected 'Bert' and '0.3' in the file).")
        print("✅ [PASS] CSV content verifies valid JOIN (Names + PageRank).")

    @classmethod
    def tearDownClass(cls):
        # Optional: Clean up test files
        # os.remove(TEST_PR_FILE)
        # os.remove(TEST_NAMES_FILE)
        # shutil.rmtree(OUTPUT_DIR)
        print("\n🔵 [TEARDOWN] Test Complete. (Test files left on disk for inspection)")

if __name__ == "__main__":
    # Custom test runner to control output formatting
    print("==========================================")
    print("   TASK 3 AUTOMATED CHECKLIST TESTER      ")
    print("==========================================\n")
    unittest.main(verbosity=0)