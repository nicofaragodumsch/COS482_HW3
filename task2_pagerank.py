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
    # INITIALIZE SPARK
    # ==========================================
    # Windows-friendly configuration
    conf = SparkConf().setAppName("PageRank").setMaster("local[*]")
    
    # Suppress excessive logging
    try:
        sc = SparkContext(conf=conf)
        sc.setLogLevel("ERROR")
    except Exception as e:
        print(f"Error initializing Spark: {e}")
        print("If running on Windows, make sure HADOOP_HOME is set or use 'local[*]' mode")
        sys.exit(1)
    
    print(f"📖 Reading input from: {INPUT_FILE}")
    print(f"📝 Output will be written to: {OUTPUT_FILE}")
    
    # ==========================================
    # STEP 0: READ INPUT AND BUILD GRAPH
    # ==========================================
    
    # Read edges from input file
    # Each line: "source destination"
    try:
        lines = sc.textFile(INPUT_FILE)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sc.stop()
        sys.exit(1)
    
    # Parse edges: (source, destination)
    edges = lines.map(lambda line: tuple(line.strip().split())).cache()
    
    # Get all unique vertices
    sources = edges.map(lambda edge: edge[0])
    destinations = edges.map(lambda edge: edge[1])
    all_vertices = sources.union(destinations).distinct().collect()
    all_vertices_set = set(all_vertices)
    num_vertices = len(all_vertices_set)
    
    print(f"🔢 Total vertices in graph: {num_vertices}")
    print(f"   Vertices: {sorted(all_vertices)}")
    
    # Build adjacency list: (vertex, [list of neighbors])
    # Group edges by source vertex
    adjacency_list = edges.groupByKey().mapValues(list)
    
    # For vertices with no outgoing edges, we need to add edges to all OTHER vertices
    # First, find vertices with no outgoing edges
    vertices_with_edges = adjacency_list.keys().collect()
    vertices_with_no_edges = all_vertices_set - set(vertices_with_edges)
    
    print(f"🔍 Vertices with no outgoing edges: {sorted(vertices_with_no_edges)}")
    
    # For each vertex with no outgoing edges, add edges to all other vertices
    if vertices_with_no_edges:
        # Create edges to all other vertices (excluding the vertex itself)
        no_edge_vertices_rdd = sc.parallelize([
            (vertex, [v for v in all_vertices if v != vertex])
            for vertex in vertices_with_no_edges
        ])
        # Union with existing adjacency list
        adjacency_list = adjacency_list.union(no_edge_vertices_rdd)
    
    # Ensure all vertices are in the adjacency list (even isolated ones)
    # Create RDD with empty lists for any missing vertices
    all_vertices_rdd = sc.parallelize([(v, []) for v in all_vertices])
    adjacency_list = adjacency_list.union(all_vertices_rdd).reduceByKey(lambda a, b: a if a else b)
    
    adjacency_list = adjacency_list.cache()
    
    # ==========================================
    # STEP 1: INITIALIZE PAGE RANKS
    # ==========================================
    
    # Initialize all page ranks to 1.0
    ranks = sc.parallelize([(vertex, 1.0) for vertex in all_vertices])
    
    print(f"\n🚀 Starting PageRank algorithm with {NUM_ITERATIONS} iterations...")
    
    # ==========================================
    # STEPS 2-4: ITERATIVE PAGERANK COMPUTATION
    # ==========================================
    
    for iteration in range(NUM_ITERATIONS):
        # Step 2: Calculate contributions
        # Each vertex contributes rank(v) / |neighbors(v)| to each of its neighbors
        
        # Join ranks with adjacency list to get (vertex, (rank, [neighbors]))
        ranks_with_neighbors = adjacency_list.join(ranks)
        
        # Calculate contributions: 
        # For each vertex, emit (neighbor, contribution) for each neighbor
        # Also emit (vertex, self_contribution) for staying on the page
        contributions = ranks_with_neighbors.flatMap(
            lambda vertex_data: compute_contributions(vertex_data, num_vertices)
        )
        
        # Step 3: Aggregate contributions and update ranks
        # Sum all contributions for each vertex
        # Formula: 0.15 + 0.85 × (sum of contributions from incoming edges)
        
        # Group contributions by vertex and sum them
        summed_contributions = contributions.reduceByKey(lambda a, b: a + b)
        
        # Update ranks using the formula
        # Note: The 0.15 already includes the random teleport (0.10) and 
        # self-loop contribution (0.05) as per the algorithm description
        ranks = summed_contributions.mapValues(lambda contrib: 0.15 + 0.85 * contrib)
        
        # Ensure all vertices have a rank (in case some have no incoming contributions)
        all_vertices_with_base = sc.parallelize([(v, 0.15) for v in all_vertices])
        ranks = ranks.union(all_vertices_with_base).reduceByKey(lambda a, b: a if a != 0.15 else b)
        
        ranks = ranks.cache()
        
        if (iteration + 1) % 5 == 0:
            print(f"   Iteration {iteration + 1}/{NUM_ITERATIONS} complete")
    
    # ==========================================
    # STEP 5: NORMALIZE BY TOTAL NUMBER OF VERTICES
    # ==========================================
    
    # Divide each rank by the total number of vertices
    final_ranks = ranks.mapValues(lambda rank: rank / num_vertices)
    
    # Sort by vertex for consistent output
    final_ranks_sorted = final_ranks.sortByKey().collect()
    
    # ==========================================
    # OUTPUT RESULTS
    # ==========================================
    
    print(f"\n✅ PageRank computation complete!")
    print(f"\n📊 Final PageRank scores:")
    for vertex, rank in final_ranks_sorted:
        print(f"   Vertex {vertex}: {rank:.6f}")
    
    # Write to output file
    output_lines = [f"{vertex} {rank}" for vertex, rank in final_ranks_sorted]
    output_rdd = sc.parallelize(output_lines)
    
    # Remove output directory if it exists
    import shutil
    if os.path.exists(OUTPUT_FILE):
        if os.path.isdir(OUTPUT_FILE):
            shutil.rmtree(OUTPUT_FILE)
        else:
            os.remove(OUTPUT_FILE)
    
    output_rdd.coalesce(1).saveAsTextFile(OUTPUT_FILE)
    
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
    # Note: If no neighbors exist, those edges were already added to all other vertices
    # in the adjacency list construction step
    
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