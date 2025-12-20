from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
import os
import shutil
import sys

# ==========================================
# CONFIGURATION
# ==========================================
# Input file from Task 2 output
PAGERANK_INPUT = "pagerank_output.txt" 
# New input file for Task 3d (Names)
NAMES_INPUT = "names.txt"
# Final output directory for Task 3e
OUTPUT_CSV_DIR = "task3_output_csv"

def create_dummy_data_if_missing():
    """
    Helper function to generate the input files if they don't exist,
    so the script runs successfully for demonstration purposes.
    """
    # 1. Generate dummy PageRank output if Task 2 wasn't run
    if not os.path.exists(PAGERANK_INPUT):
        print(f"⚠️ {PAGERANK_INPUT} not found. Creating dummy data based on Task 2 example...")
        with open("pagerank_dummy_data.txt", "w") as f:
            # Example values approximating the PDF example
            f.write("0 0.179\n")
            f.write("1 0.214\n")
            f.write("2 0.285\n")
            f.write("3 0.321\n")
        return "pagerank_dummy_data.txt"
    return PAGERANK_INPUT

    # 2. Generate names.txt as specified in Task 3d [cite: 120-123]
    if not os.path.exists(NAMES_INPUT):
        print(f"Creating {NAMES_INPUT}...")
        with open(NAMES_INPUT, "w") as f:
            f.write("0 Adam\n")
            f.write("1 Lisa\n")
            f.write("2 Bert\n")
            f.write("3 Ralph\n")

def main():
    # Ensure input data exists
    pr_input_file = create_dummy_data_if_missing()
    
    # Create names.txt if it doesn't exist (Required for Step D)
    if not os.path.exists(NAMES_INPUT):
        with open(NAMES_INPUT, "w") as f:
            f.write("0 Adam\n1 Lisa\n2 Bert\n3 Ralph\n")

    # ==========================================
    # INITIALIZE SPARK SESSION
    # ==========================================
    spark = (SparkSession.builder
             .appName("Task3_SparkSQL")
             .master("local[1]") 
             .getOrCreate())
    
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    print("\n=== Task 3: Spark SQL ===\n")

    # ==========================================
    # TASK 3(a): Create DataFrame from PageRank RDD
    # ==========================================
    # "Create a Spark data frame from the RDD containing the page ranks... computed in Task 2." [cite: 114]
    
    print(f"Reading PageRank data from: {pr_input_file}")
    
    # Read text file as RDD
    pr_rdd_raw = sc.textFile(pr_input_file)
    
    # Parse RDD: "0 0.15" -> (0, 0.15)
    # Assuming space-separated values from Task 2 output
    pr_rdd_parsed = pr_rdd_raw.map(lambda line: line.strip().split())\
                              .map(lambda parts: (int(parts[0]), float(parts[1])))
    
    # Define Schema: id, page_rank
    pr_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("page_rank", FloatType(), True)
    ])
    
    # Create DataFrame
    pr_df = spark.createDataFrame(pr_rdd_parsed, schema=pr_schema)
    
    # Register as Temporary View for SQL queries
    pr_df.createOrReplaceTempView("pageranks")
    
    print("✅ PageRank DataFrame created and view 'pageranks' registered.")
    pr_df.show()

    # ==========================================
    # TASK 3(b): Find PageRank of Vertex 2
    # ==========================================
    # "Write a Spark SQL query to find the page rank of vertex 2. Print it to the screen." [cite: 115]
    
    print("\n--- Task 3(b): PageRank of Vertex 2 ---")
    query_b = spark.sql("SELECT page_rank FROM pageranks WHERE id = 2")
    
    # Show result
    query_b.show()

    # ==========================================
    # TASK 3(c): Find Vertex with Largest PageRank
    # ==========================================
    # "Write a Spark SQL query to find the vertex with the largest page rank. Print both the vertex ID and its page rank." [cite: 116-117]
    
    print("\n--- Task 3(c): Vertex with Largest PageRank ---")
    query_c = spark.sql("SELECT id, page_rank FROM pageranks ORDER BY page_rank DESC LIMIT 1")
    
    # Show result
    query_c.show()

    # ==========================================
    # TASK 3(d): Create DataFrame from Names File
    # ==========================================
    # "Create a Spark data frame from the input text file... Schema should be: id, name." [cite: 124-125]
    
    print(f"\n--- Task 3(d): Loading Names from {NAMES_INPUT} ---")
    
    names_rdd_raw = sc.textFile(NAMES_INPUT)
    
    # Parse RDD: "0 Adam" -> (0, "Adam")
    names_rdd_parsed = names_rdd_raw.map(lambda line: line.strip().split())\
                                    .map(lambda parts: (int(parts[0]), parts[1]))
    
    names_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True)
    ])
    
    names_df = spark.createDataFrame(names_rdd_parsed, schema=names_schema)
    names_df.createOrReplaceTempView("names")
    
    names_df.show()

    # ==========================================
    # TASK 3(e): Join and Save as CSV
    # ==========================================
    # "Write a Spark SQL query to join the data frame... Save the query result in CSV format." [cite: 126-127]
    
    print("\n--- Task 3(e): Joining Tables and Saving to CSV ---")
    
    join_query = """
        SELECT n.id, n.name, p.page_rank 
        FROM pageranks p 
        JOIN names n ON p.id = n.id
    """
    
    result_df = spark.sql(join_query)
    
    print("Joined Data Preview:")
    result_df.show()
    
    # Cleanup previous output
    if os.path.exists(OUTPUT_CSV_DIR):
        if os.path.isdir(OUTPUT_CSV_DIR):
            shutil.rmtree(OUTPUT_CSV_DIR)
        else:
            os.remove(OUTPUT_CSV_DIR)
            
    # Save to CSV
    # header=True is optional but good practice for CSVs
    result_df.write.option("header", "true").csv(OUTPUT_CSV_DIR)
    
    print(f"✅ Result saved successfully to directory: {OUTPUT_CSV_DIR}")
    
    spark.stop()

if __name__ == "__main__":
    main()