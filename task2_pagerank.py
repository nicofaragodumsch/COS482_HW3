import sys
from pyspark import SparkConf, SparkContext

# ==========================================
# CONFIGURATION
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
    conf = SparkConf().setAppName("Task2_PageRank")
    sc = SparkContext(conf=conf)

    # 1. Load Data
    lines = sc.textFile(INPUT_FILE)
    
    # 2. Parse Initial Edges: (source, destination)
    # Filter empty lines if any
    original_edges = lines.filter(lambda x: len(x.strip()) > 0).map(parse_edge)
    
    # Cache edges as we might need them multiple times for graph construction
    original_edges.cache()

    # 3. Identify all distinct vertices
    # We look at both sources and destinations to find every unique node
    distinct_vertices = original_edges.flatMap(lambda x: x).distinct()
    vertex_count = distinct_vertices.count()
    
    # 4. Handle Sink Nodes (Assignment Step 0)
    # "If a node has no outgoing edge, add an edge from the node to every other node"
    
    # Find sources (nodes that have outgoing edges)
    sources = original_edges.map(lambda x: x[0]).distinct()
    
    # Find sinks (vertices - sources)
    sinks = distinct_vertices.subtract(sources)
    
    # Create new edges for sinks: (sink, other_node) where sink != other_node
    # We use cartesian product of sinks and all vertices, then filter out self-loops
    new_sink_edges = sinks.cartesian(distinct_vertices).filter(lambda x: x[0] != x[1])
    
    # Combine original edges with the newly created sink edges
    all_edges = original_edges.union(new_sink_edges)
    
    # Group by source to create the adjacency list: (source, [dest1, dest2, ...])
    # cache() this because the graph topology doesn't change during iteration
    adjacency_list = all_edges.groupByKey().mapValues(list).cache()

    # 5. Initialize Ranks (Assignment Step 1)
    # "Initialize the page rank of every node as 1."
    ranks = distinct_vertices.map(lambda v: (v, 1.0))

    # 6. Run PageRank Iterations (Assignment Step 4)
    # "Repeat steps 2 and 3 k times (you may set k=10)."
    for i in range(ITERATIONS):
        # Join graph structure with current ranks:
        # Result: (node, ([neighbors], rank))
        # Note: We perform a join on adjacency_list. Since we fixed sink nodes,
        # EVERY node is now a source, so a standard join covers all vertices.
        contribs = adjacency_list.join(ranks).flatMap(compute_contributions)
        
        # Sum contributions by destination
        # Result: (node, sum_of_contributions)
        total_contribs = contribs.reduceByKey(lambda x, y: x + y)
        
        # Update Ranks (Assignment Step 3)
        # "Set each vertex's rank to 0.15 + 0.85 * (contributions)"
        # Note: We use distinct_vertices.leftOuterJoin to ensure nodes that received 
        # ZERO contributions (no incoming edges) still get included with a default sum of 0.0.
        ranks = distinct_vertices.map(lambda v: (v, 0.0)) \
            .leftOuterJoin(total_contribs) \
            .mapValues(lambda x: 0.15 + 0.85 * (x[1] if x[1] is not None else 0.0))

    # 7. Normalize (Assignment Step 5)
    # "Divide the page rank of every vertex by the total number of vertices"
    final_ranks = ranks.mapValues(lambda rank: rank / vertex_count)

    # 8. Output results
    # Format: "vertex_id <rank>"
    # coalese(1) allows us to save as a single part file for easier reading (optional optimization)
    final_ranks.map(lambda x: f"{x[0]} {x[1]}") \
               .coalesce(1) \
               .saveAsTextFile(OUTPUT_DIR)

    print(f"PageRank complete. Output saved to directory: {OUTPUT_DIR}")
    
    sc.stop()

if __name__ == "__main__":
    main()