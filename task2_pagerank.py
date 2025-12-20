from pyspark import SparkContext, SparkConf
import sys
import os

def main():
    """
    Task 2: PageRank Implementation using Spark RDD API
    
    This implements the simple PageRank algorithm with the random surfer model:
    - Stay on page: 5%
    - Follow a link: 85%
    - Random teleport: 10%
    """
    
    # ==========================================
    # CONFIGURATION
    # ==========================================
    INPUT_FILE = "pagerank_input.txt"  # Default input file
    OUTPUT_FILE = "pagerank_output.txt"  # Default output file
    NUM_ITERATIONS = 10  # Number of iterations (k=10 as specified)
    
    # Allow command line arguments for input/output files
    if len(sys.argv) > 1:
        INPUT_FILE = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_FILE = sys.argv[2]
    
    # ==========================================
    # INITIALIZE SPARK (Windows-optimized)
    # ==========================================
    conf = (SparkConf()
            .setAppName("PageRank")
            .setMaster("local[1]")  # Use only 1 thread to avoid connection issues
            .set("spark.driver.host", "localhost")
            .set("spark.driver.bindAddress", "127.0.0.1")
            .set("spark.ui.enabled", "false")  # Disable UI
            .set("spark.sql.shuffle.partitions", "1"))  # Minimal shuffling
    
    try:
        sc = SparkContext(conf=conf)
        sc.setLogLevel("ERROR")
    except Exception as e:
        print(f"Error initializing Spark: {e}")
        sys.exit(1)
    
    print(f"📖 Reading input from: {INPUT_FILE}")
    print(f"📝 Output will be written to: {OUTPUT_FILE}")
    
    # ==========================================
    # STEP 0: READ INPUT AND BUILD GRAPH
    # ==========================================
    
    try:
        lines = sc.textFile(INPUT_FILE)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sc.stop()
        sys.exit(1)
    
    # Parse edges: (source, destination) - NO CACHE to avoid serialization issues
    edges = lines.map(lambda line: tuple(line.strip().split()))
    
    # Get all unique vertices - collect early to avoid issues
    edges_list = edges.collect()  # Collect to driver to avoid Java communication issues
    
    all_vertices_set = set()
    for src, dst in edges_list:
        all_vertices_set.add(src)
        all_vertices_set.add(dst)
    
    all_vertices = sorted(list(all_vertices_set))
    num_vertices = len(all_vertices)
    
    print(f"🔢 Total vertices in graph: {num_vertices}")
    print(f"   Vertices: {all_vertices}")
    
    # Build adjacency list in Python (not Spark) to avoid issues
    adjacency_dict = {v: [] for v in all_vertices}
    for src, dst in edges_list:
        adjacency_dict[src].append(dst)
    
    # For vertices with no outgoing edges, add edges to all OTHER vertices
    vertices_with_no_edges = [v for v in all_vertices if len(adjacency_dict[v]) == 0]
    
    print(f"🔍 Vertices with no outgoing edges: {vertices_with_no_edges}")
    
    for vertex in vertices_with_no_edges:
        adjacency_dict[vertex] = [v for v in all_vertices if v != vertex]
    
    # Convert back to RDD
    adjacency_list = sc.parallelize(list(adjacency_dict.items()))
    
    # ==========================================
    # STEP 1: INITIALIZE PAGE RANKS
    # ==========================================
    
    ranks = sc.parallelize([(vertex, 1.0) for vertex in all_vertices])
    
    print(f"\n🚀 Starting PageRank algorithm with {NUM_ITERATIONS} iterations...")
    
    # ==========================================
    # STEPS 2-4: ITERATIVE PAGERANK COMPUTATION
    # ==========================================
    
    for iteration in range(NUM_ITERATIONS):
        # Join ranks with adjacency list
        ranks_with_neighbors = adjacency_list.join(ranks)
        
        # Calculate contributions
        contributions = ranks_with_neighbors.flatMap(
            lambda vertex_data: compute_contributions(vertex_data, num_vertices)
        )
        
        # Sum contributions
        summed_contributions = contributions.reduceByKey(lambda a, b: a + b)
        
        # Update ranks
        ranks = summed_contributions.mapValues(lambda contrib: 0.15 + 0.85 * contrib)
        
        # Ensure all vertices have ranks
        all_vertices_with_base = sc.parallelize([(v, 0.15) for v in all_vertices])
        ranks = ranks.union(all_vertices_with_base).reduceByKey(lambda a, b: a if a != 0.15 else b)
        
        if (iteration + 1) % 5 == 0:
            print(f"   Iteration {iteration + 1}/{NUM_ITERATIONS} complete")
    
    # ==========================================
    # STEP 5: NORMALIZE BY TOTAL NUMBER OF VERTICES
    # ==========================================
    
    final_ranks = ranks.mapValues(lambda rank: rank / num_vertices)
    
    # Collect results to driver
    final_ranks_sorted = sorted(final_ranks.collect(), key=lambda x: x[0])
    
    # ==========================================
    # OUTPUT RESULTS
    # ==========================================
    
    print(f"\n✅ PageRank computation complete!")
    print(f"\n📊 Final PageRank scores:")
    for vertex, rank in final_ranks_sorted:
        print(f"   Vertex {vertex}: {rank:.6f}")
    
    # Write to output file
    output_lines = [f"{vertex} {rank}" for vertex, rank in final_ranks_sorted]
    output_rdd = sc.parallelize(output_lines, 1)  # Single partition
    
    # Remove output directory if it exists
    import shutil
    if os.path.exists(OUTPUT_FILE):
        if os.path.isdir(OUTPUT_FILE):
            shutil.rmtree(OUTPUT_FILE)
        else:
            os.remove(OUTPUT_FILE)
    
    output_rdd.saveAsTextFile(OUTPUT_FILE)
    
    print(f"\n💾 Results saved to: {OUTPUT_FILE}")
    print(f"   (Output is in a directory - check {OUTPUT_FILE}/part-00000)")
    
    # Stop Spark context
    sc.stop()


def compute_contributions(vertex_data, num_vertices):
    """
    Compute contributions from a vertex to its neighbors.
    
    Args:
        vertex_data: Tuple of (vertex, (neighbors_list, rank))
        num_vertices: Total number of vertices in the graph
    
    Returns:
        List of (target_vertex, contribution) tuples
    
    The random surfer model:
    - 5% chance to stay on current page (self-loop)
    - 85% chance to follow a link (distributed among neighbors)
    - 10% chance to randomly teleport (handled separately, not per-vertex)
    """
    vertex, (neighbors, rank) = vertex_data
    contributions = []
    
    num_neighbors = len(neighbors)
    
    # Contribution 1: Stay on the page (5% of rank)
    self_contribution = 0.05 * rank
    contributions.append((vertex, self_contribution))
    
    # Contribution 2: Follow links (85% of rank distributed among neighbors)
    if num_neighbors > 0:
        contribution_per_neighbor = (0.85 * rank) / num_neighbors
        for neighbor in neighbors:
            contributions.append((neighbor, contribution_per_neighbor))
    
    # Contribution 3: Random teleport (10% of ALL ranks distributed to ALL vertices)
    # This is handled implicitly in the rank update formula:
    # The 0.15 in "0.15 + 0.85 × contributions" accounts for:
    # - 0.10 from random teleport (distributed from all vertices)
    # - 0.05 from self-loop (already added above)
    # Since sum of all ranks is num_vertices (each initialized to 1.0),
    # each vertex gets 0.10 * num_vertices / num_vertices = 0.10 from teleport
    
    return contributions


if __name__ == "__main__":
    main()