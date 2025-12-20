import sys
import os

# ==============================================================================
# 💀 EXTREME WINDOWS PATCHING (The "PYTHONSTARTUP" Fix)
# ==============================================================================
# We create a temporary python file that patches 'socketserver'.
# Then we set the PYTHONSTARTUP environment variable to point to it.
# This forces EVERY Python process (driver AND workers) to run this patch 
# immediately on startup, before PySpark even loads.

patch_filename = "fix_spark_windows.py"
patch_content = """
import sys
import os
import socketserver

# FIX 1: Patch socketserver.UnixStreamServer for Windows
# This class was removed in Python 3 on Windows, but PySpark needs it.
if sys.platform == 'win32':
    if not hasattr(socketserver, 'UnixStreamServer'):
        socketserver.UnixStreamServer = socketserver.TCPServer

# FIX 2: Force localhost binding to avoid firewall/VPN issues
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
"""

# Write the patch file to disk
with open(patch_filename, "w") as f:
    f.write(patch_content)

# Tell Python to run this file on startup for ALL processes
os.environ["PYTHONSTARTUP"] = os.path.abspath(patch_filename)

# Also apply the patch to the current process manually, just in case
import socketserver
if sys.platform == 'win32':
    if not hasattr(socketserver, 'UnixStreamServer'):
        socketserver.UnixStreamServer = socketserver.TCPServer

# ==============================================================================
# SPARK SETUP
# ==============================================================================
# Force Spark to use the current Python executable
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark import SparkConf, SparkContext

# ==============================================================================
# TASK 2 LOGIC
# ==============================================================================
INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else 'graph.txt'
OUTPUT_DIR = 'task2_output'
ITERATIONS = 10

def parse_edge(line):
    """Parses a line 'source destination'."""
    parts = line.strip().split()
    return (parts[0], parts[1])

def compute_contributions(node_data):
    """
    node_data: (node_id, ([neighbors], rank))
    Yields: (neighbor, contribution)
    """
    neighbors = node_data[1][0]
    rank = node_data[1][1]
    num_neighbors = len(neighbors)
    
    for neighbor in neighbors:
        yield (neighbor, rank / num_neighbors)

def main():
    # 1. Initialize Spark
    # Use 'local[1]' to run on a single thread (Safest for Windows)
    conf = SparkConf() \
        .setAppName("Task2_PageRank") \
        .setMaster("local[1]") \
        .set("spark.driver.bindAddress", "127.0.0.1")
    
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    try:
        # 2. Load and Parse Data
        lines = sc.textFile(INPUT_FILE)
        
        # Filter empty lines and parse
        original_edges = lines.filter(lambda x: len(x.strip()) > 0).map(parse_edge)
        original_edges.cache()

        # 3. Identify all vertices (Step 0 Part A)
        distinct_vertices = original_edges.flatMap(lambda x: x).distinct()
        
        # Trigger action to verify graph loaded (and worker connectivity)
        N = distinct_vertices.count()
        print(f"DEBUG: Graph loaded. Vertices: {N}")

        # 4. Handle Sink Nodes (Step 0 Part B)
        # "Add edge from sink to every other node"
        sources = original_edges.map(lambda x: x[0]).distinct()
        sinks = distinct_vertices.subtract(sources)
        
        # Cartesian product to create edges from every sink to every other node
        # Filter x[0] != x[1] to avoid self-loops if implied by instructions
        new_sink_edges = sinks.cartesian(distinct_vertices).filter(lambda x: x[0] != x[1])
        
        # Combine
        all_edges = original_edges.union(new_sink_edges)
        
        # Create Adjacency List
        adjacency_list = all_edges.groupByKey().mapValues(list).cache()

        # 5. Initialize Ranks (Step 1) -> 1.0
        ranks = distinct_vertices.map(lambda v: (v, 1.0))

        # 6. Iterations (Step 4)
        for i in range(ITERATIONS):
            # Join graph with ranks
            contribs = adjacency_list.join(ranks).flatMap(compute_contributions)
            
            # Sum contributions
            total_contribs = contribs.reduceByKey(lambda x, y: x + y)
            
            # Update Ranks (Step 3) -> 0.15 + 0.85 * contributions
            # leftOuterJoin handles nodes that received 0 contributions
            ranks = distinct_vertices.map(lambda v: (v, 0.0)) \
                .leftOuterJoin(total_contribs) \
                .mapValues(lambda x: 0.15 + 0.85 * (x[1] if x[1] is not None else 0.0))

        # 7. Normalize (Step 5)
        final_ranks = ranks.mapValues(lambda rank: rank / N)

        # 8. Save Output
        if os.path.exists(OUTPUT_DIR):
            import shutil
            shutil.rmtree(OUTPUT_DIR)

        final_ranks.map(lambda x: f"{x[0]} {x[1]}") \
                   .coalesce(1) \
                   .saveAsTextFile(OUTPUT_DIR)

        print(f"PageRank complete. Saved to: {OUTPUT_DIR}")

    except Exception as e:
        print(f"❌ Spark Error: {e}")
        # Delete the patch file on exit if you want cleanup, 
        # but keeping it is fine for debugging.
        sys.exit(1)
    finally:
        sc.stop()

if __name__ == "__main__":
    main()