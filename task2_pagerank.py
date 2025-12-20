import sys
import os

# ==========================================
# STEP 1: GLOBAL WORKER PATCH (The Fix)
# ==========================================
# We create a 'usercustomize.py' file. Python imports this automatically 
# on startup. This ensures that every Spark worker process gets patched 
# BEFORE it tries to import PySpark.

patch_code = """
import sys
import socketserver

# CRITICAL WINDOWS FIX:
# PySpark workers on Windows try to use UnixStreamServer, which is missing.
# We map it to TCPServer to prevent the 'AttributeError' crash.
if sys.platform == "win32":
    if not hasattr(socketserver, 'UnixStreamServer'):
        socketserver.UnixStreamServer = socketserver.TCPServer
"""

# Write the patch file to the current directory
with open("usercustomize.py", "w") as f:
    f.write(patch_code)

# Add current directory to PYTHONPATH so workers pick up the usercustomize.py
current_dir = os.getcwd()
if "PYTHONPATH" in os.environ:
    os.environ["PYTHONPATH"] = current_dir + os.pathsep + os.environ["PYTHONPATH"]
else:
    os.environ["PYTHONPATH"] = current_dir

# ==========================================
# STEP 2: ENVIRONMENT CONFIGURATION
# ==========================================

# 1. Fix "Python worker failed to connect back"
# Ensure workers use the exact same Python executable as the driver.
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# 2. Fix Hadoop Binary Warning (Optional but good)
# If you created C:\hadoop\bin\winutils.exe, this helps Spark find it.
if os.path.exists("C:\\hadoop"):
    os.environ['HADOOP_HOME'] = "C:\\hadoop"

# 3. Force Localhost Networking
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'

# Now it is safe to import PySpark
from pyspark import SparkConf, SparkContext

# ==========================================
# STEP 3: MAIN LOGIC (PageRank)
# ==========================================
INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else 'graph.txt'
OUTPUT_DIR = 'task2_output'
ITERATIONS = 10

def parse_edge(line):
    parts = line.strip().split()
    return (parts[0], parts[1])

def compute_contributions(node_data):
    neighbors = node_data[1][0]
    rank = node_data[1][1]
    num_neighbors = len(neighbors)
    for neighbor in neighbors:
        yield (neighbor, rank / num_neighbors)

def main():
    # Initialize Spark
    # local[1] is safest for Windows to avoid multi-thread networking issues
    conf = SparkConf() \
        .setAppName("Task2_PageRank") \
        .setMaster("local[1]") \
        .set("spark.driver.host", "127.0.0.1") \
        .set("spark.driver.bindAddress", "127.0.0.1")
    
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    try:
        # 1. Load Data
        lines = sc.textFile(INPUT_FILE)
        
        # 2. Parse Edges
        original_edges = lines.filter(lambda x: len(x.strip()) > 0).map(parse_edge)
        original_edges.cache()

        # 3. Distinct Vertices
        distinct_vertices = original_edges.flatMap(lambda x: x).distinct()
        
        # Action to verify load (and test if workers are alive)
        vertex_count = distinct_vertices.count()
        print(f"DEBUG: Graph loaded. Found {vertex_count} unique vertices.")

        # 4. Handle Sinks (Step 0)
        sources = original_edges.map(lambda x: x[0]).distinct()
        sinks = distinct_vertices.subtract(sources)
        new_sink_edges = sinks.cartesian(distinct_vertices).filter(lambda x: x[0] != x[1])
        all_edges = original_edges.union(new_sink_edges)
        
        adjacency_list = all_edges.groupByKey().mapValues(list).cache()

        # 5. Initialize Ranks (Step 1)
        ranks = distinct_vertices.map(lambda v: (v, 1.0))

        # 6. PageRank Loop (Step 4)
        for i in range(ITERATIONS):
            contribs = adjacency_list.join(ranks).flatMap(compute_contributions)
            total_contribs = contribs.reduceByKey(lambda x, y: x + y)
            ranks = distinct_vertices.map(lambda v: (v, 0.0)) \
                .leftOuterJoin(total_contribs) \
                .mapValues(lambda x: 0.15 + 0.85 * (x[1] if x[1] is not None else 0.0))

        # 7. Normalize (Step 5)
        final_ranks = ranks.mapValues(lambda rank: rank / vertex_count)

        # 8. Output
        if os.path.exists(OUTPUT_DIR):
            import shutil
            shutil.rmtree(OUTPUT_DIR)

        final_ranks.map(lambda x: f"{x[0]} {x[1]}") \
                   .coalesce(1) \
                   .saveAsTextFile(OUTPUT_DIR)

        print(f"PageRank complete. Output saved to: {OUTPUT_DIR}")

    except Exception as e:
        print(f"❌ Spark Execution Error: {e}")
        sys.exit(1)
    finally:
        sc.stop()

if __name__ == "__main__":
    main()