import sys
import os
import socketserver

# ==============================================================================
# 💀 CRITICAL WINDOWS PATCH (MUST BE AT THE VERY TOP)
# ==============================================================================
# This specific block MUST run before 'from pyspark import ...'
# otherwise, Python 3.14 on Windows will crash immediately with an AttributeError.

if sys.platform == "win32":
    # 1. Patch socketserver.UnixStreamServer
    # PySpark expects this class to exist, but it was removed in Windows Python.
    # We map it to TCPServer so PySpark can load without crashing.
    if not hasattr(socketserver, 'UnixStreamServer'):
        socketserver.UnixStreamServer = socketserver.TCPServer

    # 2. Fix "Python worker failed to connect back"
    # Ensure Spark uses the exact same Python executable as this script.
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    # 3. Force Localhost Networking
    # Prevents VPN/Firewall/IPv6 issues from blocking the worker connection.
    os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'

# ==============================================================================
# NOW IT IS SAFE TO IMPORT PYSPARK
# ==============================================================================
from pyspark import SparkConf, SparkContext

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else 'graph.txt'
OUTPUT_DIR = 'task2_output'
ITERATIONS = 10

def parse_edge(line):
    """Parses a line 'source destination' into a tuple."""
    parts = line.strip().split()
    return (parts[0], parts[1])

def compute_contributions(node_data):
    """
    Calculates (destination, contribution) for a given node.
    node_data is: (node_id, ([list_of_neighbors], current_rank))
    """
    neighbors = node_data[1][0]
    rank = node_data[1][1]
    num_neighbors = len(neighbors)
    
    for neighbor in neighbors:
        yield (neighbor, rank / num_neighbors)

def main():
    # Initialize Spark
    # local[1] forces single-threaded execution, which is the most stable
    # configuration for PySpark on Windows without Hadoop binaries.
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
        
        # 2. Parse Initial Edges: (source, destination)
        original_edges = lines.filter(lambda x: len(x.strip()) > 0).map(parse_edge)
        
        # Cache edges as we might need them multiple times for graph construction
        original_edges.cache()

        # 3. Identify all distinct vertices
        distinct_vertices = original_edges.flatMap(lambda x: x).distinct()
        
        # Action: Trigger data load to verify everything is working
        vertex_count = distinct_vertices.count()
        print(f"DEBUG: Graph loaded. Found {vertex_count} unique vertices.")

        # 4. Handle Sink Nodes (Assignment Step 0)
        # "If a node has no outgoing edge, add an edge from the node to every other node"
        
        sources = original_edges.map(lambda x: x[0]).distinct()
        sinks = distinct_vertices.subtract(sources)
        
        # Create new edges for sinks
        new_sink_edges = sinks.cartesian(distinct_vertices).filter(lambda x: x[0] != x[1])
        
        # Combine original edges with the newly created sink edges
        all_edges = original_edges.union(new_sink_edges)
        
        # Group by source to create the adjacency list
        adjacency_list = all_edges.groupByKey().mapValues(list).cache()

        # 5. Initialize Ranks (Assignment Step 1)
        ranks = distinct_vertices.map(lambda v: (v, 1.0))

        # 6. Run PageRank Iterations (Assignment Step 4)
        for i in range(ITERATIONS):
            # Join graph structure with current ranks
            contribs = adjacency_list.join(ranks).flatMap(compute_contributions)
            
            # Sum contributions by destination
            total_contribs = contribs.reduceByKey(lambda x, y: x + y)
            
            # Update Ranks (Assignment Step 3)
            # 0.15 + 0.85 * (contributions)
            ranks = distinct_vertices.map(lambda v: (v, 0.0)) \
                .leftOuterJoin(total_contribs) \
                .mapValues(lambda x: 0.15 + 0.85 * (x[1] if x[1] is not None else 0.0))

        # 7. Normalize (Assignment Step 5)
        # "Divide the page rank of every vertex by the total number of vertices"
        final_ranks = ranks.mapValues(lambda rank: rank / vertex_count)

        # 8. Output results
        if os.path.exists(OUTPUT_DIR):
            import shutil
            shutil.rmtree(OUTPUT_DIR)

        final_ranks.map(lambda x: f"{x[0]} {x[1]}") \
                   .coalesce(1) \
                   .saveAsTextFile(OUTPUT_DIR)

        print(f"PageRank complete. Output saved to directory: {OUTPUT_DIR}")

    except Exception as e:
        print(f"❌ Error during Spark Execution: {e}")
        # Explicitly exit with error code so the test script knows it failed
        sys.exit(1)
        
    finally:
        sc.stop()

if __name__ == "__main__":
    main()