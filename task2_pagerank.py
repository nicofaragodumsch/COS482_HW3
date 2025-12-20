import sys
import os
import socketserver

# ==========================================
# WINDOWS & NETWORK FIXES (CRITICAL)
# ==========================================

# 1. Patch socketserver for Windows
#    PySpark tries to use 'UnixStreamServer' which doesn't exist on Windows.
if sys.platform == "win32":
    socketserver.UnixStreamServer = socketserver.TCPServer

# 2. Force Spark to use the CURRENT Python executable
#    This ensures the worker processes match the driver process version,
#    fixing the "Python worker failed to connect back" error.
current_python = sys.executable
os.environ['PYSPARK_PYTHON'] = current_python
os.environ['PYSPARK_DRIVER_PYTHON'] = current_python

# 3. Force Localhost Binding for Networking
#    This prevents Spark from binding to the wrong network adapter (VPN, IPv6, etc.)
#    and ensures the worker can talk back to the driver on 127.0.0.1.
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'

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
    # Initialize Spark Configuration
    # We explicitly set bind addresses to localhost to match the environment variables above.
    conf = SparkConf() \
        .setAppName("Task2_PageRank") \
        .setMaster("local[1]") \
        .set("spark.driver.host", "127.0.0.1") \
        .set("spark.driver.bindAddress", "127.0.0.1") \
        .set("spark.network.timeout", "600s") \
        .set("spark.executor.heartbeatInterval", "60s")
    
    sc = SparkContext(conf=conf)
    
    # Set log level to ERROR to reduce console noise
    sc.setLogLevel("ERROR")

    try:
        # 1. Load Data
        lines = sc.textFile(INPUT_FILE)
        
        # 2. Parse Initial Edges: (source, destination)
        original_edges = lines.filter(lambda x: len(x.strip()) > 0).map(parse_edge)
        
        # Cache edges as we use them multiple times
        original_edges.cache()

        # 3. Identify all distinct vertices
        distinct_vertices = original_edges.flatMap(lambda x: x).distinct()
        
        # Trigger an action to verify data load immediately
        vertex_count = distinct_vertices.count()
        print(f"DEBUG: Graph loaded. Found {vertex_count} unique vertices.")

        # 4. Handle Sink Nodes (Assignment Step 0)
        # "If a node has no outgoing edge, add an edge from the node to every other node"
        
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
            # Formula: 0.15 + 0.85 * (sum_of_incoming_contributions)
            # leftOuterJoin ensures nodes with 0 incoming edges still get the base score
            ranks = distinct_vertices.map(lambda v: (v, 0.0)) \
                .leftOuterJoin(total_contribs) \
                .mapValues(lambda x: 0.15 + 0.85 * (x[1] if x[1] is not None else 0.0))

        # 7. Normalize (Assignment Step 5)
        # "Divide the page rank of every vertex by the total number of vertices"
        final_ranks = ranks.mapValues(lambda rank: rank / vertex_count)

        # 8. Output results
        # Check if output dir exists and remove it (local fs only)
        if os.path.exists(OUTPUT_DIR):
            import shutil
            shutil.rmtree(OUTPUT_DIR)

        # Save output
        final_ranks.map(lambda x: f"{x[0]} {x[1]}") \
                   .coalesce(1) \
                   .saveAsTextFile(OUTPUT_DIR)

        print(f"PageRank complete. Output saved to directory: {OUTPUT_DIR}")

    except Exception as e:
        print(f"❌ Error during Spark Execution: {e}")
        # Re-raise or exit with error code so the test script knows it failed
        sys.exit(1)
        
    finally:
        sc.stop()

if __name__ == "__main__":
    main()