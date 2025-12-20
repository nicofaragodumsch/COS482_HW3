import sys
import os
import socketserver

# ==========================================
# WINDOWS SPECIFIC CONFIGURATION
# ==========================================

# 1. Patch socketserver for Windows
# PySpark internally uses UnixStreamServer, which is missing on Windows.
if sys.platform == "win32":
    socketserver.UnixStreamServer = socketserver.TCPServer

# 2. Force Spark to use the EXACT same Python interpreter as this script
# This fixes the "Python worker failed to connect back" error.
current_python = sys.executable
os.environ['PYSPARK_PYTHON'] = current_python
os.environ['PYSPARK_DRIVER_PYTHON'] = current_python

from pyspark import SparkConf, SparkContext

# ==========================================
# MAIN LOGIC
# ==========================================

# Allow input file to be passed as argument, default to 'graph.txt'
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
    # CRITICAL FIX: Changed "local[*]" to "local[1]"
    # "local[1]" runs in a single thread. This prevents the "worker failed to connect" 
    # error caused by Windows Firewall or network stack issues blocking multiple workers.
    conf = SparkConf().setAppName("Task2_PageRank").setMaster("local[1]")
    
    # Optional: Increase timeout settings to be more forgiving on slow starts
    conf.set("spark.network.timeout", "600s")
    conf.set("spark.executor.heartbeatInterval", "60s")
    
    sc = SparkContext(conf=conf)
    
    # Set log level to ERROR to reduce console noise
    sc.setLogLevel("ERROR")

    try:
        # 1. Load Data
        lines = sc.textFile(INPUT_FILE)
        
        # 2. Parse Initial Edges: (source, destination)
        original_edges = lines.filter(lambda x: len(x.strip()) > 0).map(parse_edge)
        
        # Cache edges
        original_edges.cache()

        # 3. Identify all distinct vertices
        distinct_vertices = original_edges.flatMap(lambda x: x).distinct()
        vertex_count = distinct_vertices.count()
        
        # 4. Handle Sink Nodes (Assignment Step 0)
        # Find sources (nodes that have outgoing edges)
        sources = original_edges.map(lambda x: x[0]).distinct()
        
        # Find sinks (vertices - sources)
        sinks = distinct_vertices.subtract(sources)
        
        # Create new edges for sinks: (sink, other_node) where sink != other_node
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
            # 0.15 + 0.85 * (sum of contributions)
            ranks = distinct_vertices.map(lambda v: (v, 0.0)) \
                .leftOuterJoin(total_contribs) \
                .mapValues(lambda x: 0.15 + 0.85 * (x[1] if x[1] is not None else 0.0))

        # 7. Normalize (Assignment Step 5)
        final_ranks = ranks.mapValues(lambda rank: rank / vertex_count)

        # 8. Output results
        # Clean output dir if exists (Local filesystem only)
        if os.path.exists(OUTPUT_DIR):
            import shutil
            shutil.rmtree(OUTPUT_DIR)

        final_ranks.map(lambda x: f"{x[0]} {x[1]}") \
                   .coalesce(1) \
                   .saveAsTextFile(OUTPUT_DIR)

        print(f"PageRank complete. Output saved to directory: {OUTPUT_DIR}")
    
    except Exception as e:
        print(f"Error during Spark Execution: {e}")
        raise
    finally:
        sc.stop()

if __name__ == "__main__":
    main()