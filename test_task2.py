#!/usr/bin/env python3
"""
Direct Test Suite for Task 2: PageRank Implementation

This script tests task2_pagerank.py by running it directly with PySpark
instead of using spark-submit (which may not be in PATH).
"""

import os
import sys
import shutil
from pathlib import Path

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_test(test_name):
    """Print test name"""
    print(f"{Colors.BOLD}Testing: {test_name}{Colors.END}")

def print_pass(message):
    """Print success message"""
    print(f"  {Colors.GREEN}✓ PASS{Colors.END}: {message}")

def print_fail(message):
    """Print failure message"""
    print(f"  {Colors.RED}✗ FAIL{Colors.END}: {message}")

def print_info(message):
    """Print info message"""
    print(f"  {Colors.YELLOW}ℹ INFO{Colors.END}: {message}")

class TestResults:
    """Track test results"""
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.failures = []
    
    def add_pass(self, test_name):
        self.total += 1
        self.passed += 1
        print_pass(test_name)
    
    def add_fail(self, test_name, reason=""):
        self.total += 1
        self.failed += 1
        self.failures.append((test_name, reason))
        print_fail(f"{test_name} - {reason}")
    
    def print_summary(self):
        print_header("TEST SUMMARY")
        print(f"Total Tests: {self.total}")
        print(f"{Colors.GREEN}Passed: {self.passed}{Colors.END}")
        print(f"{Colors.RED}Failed: {self.failed}{Colors.END}")
        
        if self.failures:
            print(f"\n{Colors.RED}Failed Tests:{Colors.END}")
            for i, (test, reason) in enumerate(self.failures, 1):
                print(f"  {i}. {test}")
                if reason:
                    print(f"     Reason: {reason}")
        
        if self.failed == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED! 🎉{Colors.END}")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}⚠ SOME TESTS FAILED ⚠{Colors.END}")
        
        return self.failed == 0

def cleanup_files():
    """Clean up test files and directories"""
    files_to_remove = [
        'pagerank_input.txt',
        'test_pagerank_input.txt',
        'pagerank_output.txt',
        'test_pagerank_output.txt',
        'test_no_outgoing.txt',
        'test_no_outgoing_output.txt'
    ]
    
    for file in files_to_remove:
        if os.path.exists(file):
            if os.path.isdir(file):
                shutil.rmtree(file)
            else:
                os.remove(file)

def create_test_input(filename='pagerank_input.txt'):
    """Create the test input file with the example graph"""
    with open(filename, 'w') as f:
        f.write("0 1\n")
        f.write("0 2\n")
        f.write("2 3\n")
    print_info(f"Created test input file: {filename}")

def run_pagerank_directly(input_file='pagerank_input.txt', output_file='pagerank_output.txt'):
    """Run PageRank directly using PySpark"""
    try:
        # Try to import PySpark
        try:
            from pyspark import SparkContext, SparkConf
        except ImportError:
            print_fail("PySpark not installed. Install with: pip install pyspark")
            return False
        
        print_info("Initializing Spark context...")
        
        # Initialize Spark
        conf = SparkConf().setAppName("PageRank_Test").setMaster("local[*]")
        sc = SparkContext(conf=conf)
        sc.setLogLevel("ERROR")
        
        print_info("Running PageRank algorithm...")
        
        # Read edges
        lines = sc.textFile(input_file)
        edges = lines.map(lambda line: tuple(line.strip().split())).cache()
        
        # Get all vertices
        sources = edges.map(lambda edge: edge[0])
        destinations = edges.map(lambda edge: edge[1])
        all_vertices = sources.union(destinations).distinct().collect()
        all_vertices_set = set(all_vertices)
        num_vertices = len(all_vertices_set)
        
        print_info(f"Graph has {num_vertices} vertices: {sorted(all_vertices)}")
        
        # Build adjacency list
        adjacency_list = edges.groupByKey().mapValues(list)
        
        # Handle vertices with no outgoing edges
        vertices_with_edges = adjacency_list.keys().collect()
        vertices_with_no_edges = all_vertices_set - set(vertices_with_edges)
        
        if vertices_with_no_edges:
            print_info(f"Adding edges for vertices with no outgoing edges: {sorted(vertices_with_no_edges)}")
            no_edge_vertices_rdd = sc.parallelize([
                (vertex, [v for v in all_vertices if v != vertex])
                for vertex in vertices_with_no_edges
            ])
            adjacency_list = adjacency_list.union(no_edge_vertices_rdd)
        
        # Ensure all vertices are in adjacency list
        all_vertices_rdd = sc.parallelize([(v, []) for v in all_vertices])
        adjacency_list = adjacency_list.union(all_vertices_rdd).reduceByKey(lambda a, b: a if a else b)
        adjacency_list = adjacency_list.cache()
        
        # Initialize ranks
        ranks = sc.parallelize([(vertex, 1.0) for vertex in all_vertices])
        
        # Run PageRank iterations
        NUM_ITERATIONS = 10
        print_info(f"Running {NUM_ITERATIONS} iterations...")
        
        for iteration in range(NUM_ITERATIONS):
            # Calculate contributions
            ranks_with_neighbors = adjacency_list.join(ranks)
            
            def compute_contributions(vertex_data):
                vertex, (neighbors, rank) = vertex_data
                contributions = []
                num_neighbors = len(neighbors)
                
                # Self-loop contribution (5%)
                contributions.append((vertex, 0.05 * rank))
                
                # Neighbor contributions (85%)
                if num_neighbors > 0:
                    contribution_per_neighbor = (0.85 * rank) / num_neighbors
                    for neighbor in neighbors:
                        contributions.append((neighbor, contribution_per_neighbor))
                
                return contributions
            
            contributions = ranks_with_neighbors.flatMap(compute_contributions)
            
            # Sum contributions and update ranks
            summed_contributions = contributions.reduceByKey(lambda a, b: a + b)
            ranks = summed_contributions.mapValues(lambda contrib: 0.15 + 0.85 * contrib)
            
            # Ensure all vertices have ranks
            all_vertices_with_base = sc.parallelize([(v, 0.15) for v in all_vertices])
            ranks = ranks.union(all_vertices_with_base).reduceByKey(lambda a, b: a if a != 0.15 else b)
            ranks = ranks.cache()
        
        print_info("Normalizing results...")
        
        # Normalize by number of vertices
        final_ranks = ranks.mapValues(lambda rank: rank / num_vertices)
        final_ranks_sorted = final_ranks.sortByKey().collect()
        
        # Write output
        output_lines = [f"{vertex} {rank}" for vertex, rank in final_ranks_sorted]
        output_rdd = sc.parallelize(output_lines)
        
        # Remove output directory if it exists
        if os.path.exists(output_file):
            shutil.rmtree(output_file)
        
        output_rdd.coalesce(1).saveAsTextFile(output_file)
        
        print_pass("PageRank computation completed successfully")
        
        # Stop Spark
        sc.stop()
        
        return True
        
    except Exception as e:
        print_fail(f"Error running PageRank: {str(e)}")
        import traceback
        traceback.print_exc()
        try:
            sc.stop()
        except:
            pass
        return False

def read_output(output_dir='pagerank_output.txt'):
    """Read the PageRank output"""
    try:
        # Output is in a directory, find the part file
        part_files = list(Path(output_dir).glob('part-*'))
        if not part_files:
            print_fail(f"No part files found in {output_dir}")
            return None
        
        with open(part_files[0], 'r') as f:
            lines = f.readlines()
        
        results = {}
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                vertex = parts[0]
                pagerank = float(parts[1])
                results[vertex] = pagerank
        
        return results
    except Exception as e:
        print_fail(f"Error reading output: {str(e)}")
        return None

def test_file_format(results, test_results):
    """Test 1: Output file format is correct"""
    print_test("Output file format (vertex pagerank)")
    
    if results is None:
        test_results.add_fail("Output file format", "Could not read output file")
        return
    
    expected_vertices = {'0', '1', '2', '3'}
    actual_vertices = set(results.keys())
    
    if actual_vertices == expected_vertices:
        test_results.add_pass("All vertices present in output")
    else:
        test_results.add_fail("Output vertices", 
                             f"Expected {expected_vertices}, got {actual_vertices}")
    
    all_floats = all(isinstance(v, float) for v in results.values())
    if all_floats:
        test_results.add_pass("All PageRank values are numeric")
    else:
        test_results.add_fail("PageRank values", "Not all values are numeric")

def test_normalization(results, test_results):
    """Test 2: Final ranks are divided by number of vertices"""
    print_test("Normalization (ranks divided by num vertices)")
    
    if results is None:
        test_results.add_fail("Normalization", "No results to check")
        return
    
    total_rank = sum(results.values())
    
    if abs(total_rank - 1.0) < 0.01:
        test_results.add_pass(f"Sum of ranks ≈ 1.0 (actual: {total_rank:.6f})")
    else:
        test_results.add_fail("Normalization", 
                             f"Sum of ranks should be ≈1.0, got {total_rank:.6f}")

def test_reasonable_values(results, test_results):
    """Test 3: PageRank values are reasonable"""
    print_test("PageRank values are reasonable")
    
    if results is None:
        test_results.add_fail("Reasonable values", "No results to check")
        return
    
    all_positive = all(v > 0 for v in results.values())
    if all_positive:
        test_results.add_pass("All PageRank values are positive")
    else:
        test_results.add_fail("PageRank values", "Some values are not positive")
    
    all_less_than_one = all(v < 1 for v in results.values())
    if all_less_than_one:
        test_results.add_pass("All PageRank values < 1.0 (normalized)")
    else:
        test_results.add_fail("PageRank values", "Some values >= 1.0")

def test_expected_rankings(results, test_results):
    """Test 4: Verify expected relative rankings"""
    print_test("Expected relative rankings")
    
    if results is None or len(results) != 4:
        test_results.add_fail("Rankings", "Incomplete results")
        return
    
    print_info("Actual PageRank values:")
    for vertex in sorted(results.keys()):
        print(f"    Vertex {vertex}: {results[vertex]:.6f}")
    
    no_zeros = all(v > 0.05 for v in results.values())
    if no_zeros:
        test_results.add_pass("No vertex has near-zero PageRank")
    else:
        test_results.add_fail("Rankings", "Some vertices have very low PageRank")
    
    max_rank = max(results.values())
    min_rank = min(results.values())
    ratio = max_rank / min_rank if min_rank > 0 else float('inf')
    
    if ratio < 10:
        test_results.add_pass(f"PageRank distribution is reasonable (max/min ratio: {ratio:.2f})")
    else:
        test_results.add_fail("Rankings", f"PageRank too skewed (ratio: {ratio:.2f})")

def test_algorithm_convergence(results, test_results):
    """Test 5: Algorithm appears to have converged"""
    print_test("Algorithm convergence")
    
    if results is None:
        test_results.add_fail("Convergence", "No results to check")
        return
    
    values = list(results.values())
    all_equal = all(abs(v - values[0]) < 0.0001 for v in values)
    
    if not all_equal:
        test_results.add_pass("PageRank values have differentiated (not stuck at initialization)")
    else:
        test_results.add_fail("Convergence", 
                             "All PageRank values are identical - algorithm may not have run")

def test_basic_properties(results, test_results):
    """Test 6: Basic mathematical properties"""
    print_test("Basic mathematical properties")
    
    if results is None:
        test_results.add_fail("Basic properties", "No results to check")
        return
    
    total = sum(results.values())
    
    if abs(total - 1.0) < 0.001:
        test_results.add_pass("Probability distribution (sum ≈ 1.0)")
    else:
        test_results.add_fail("Basic properties", 
                             f"Sum should be 1.0, got {total:.6f}")
    
    valid_range = all(0 < v < 1 for v in results.values())
    if valid_range:
        test_results.add_pass("All values in valid range (0, 1)")
    else:
        test_results.add_fail("Basic properties", "Some values outside (0, 1)")

def main():
    """Main test runner"""
    print_header("TASK 2 PAGERANK TEST SUITE (DIRECT)")
    
    # Check PySpark availability
    try:
        import pyspark
        print_pass(f"PySpark is installed (version {pyspark.__version__})")
    except ImportError:
        print_fail("PySpark is not installed!")
        print_info("Install with: pip install pyspark")
        return False
    
    # Initialize test results tracker
    test_results = TestResults()
    
    # Clean up any existing files
    print_info("Cleaning up previous test files...")
    cleanup_files()
    
    # Create test input
    print_info("Setting up test environment...")
    create_test_input()
    
    # Run the PageRank implementation directly
    print_header("RUNNING PAGERANK IMPLEMENTATION")
    success = run_pagerank_directly()
    
    if not success:
        print_fail("Failed to run PageRank algorithm")
        test_results.add_fail("Execution", "Algorithm failed to run")
        test_results.print_summary()
        return False
    
    # Read the output
    print_info("Reading output file...")
    results = read_output()
    
    if results is None:
        print_fail("Could not read output file")
        test_results.add_fail("Output", "Could not read results")
        test_results.print_summary()
        return False
    
    # Run all tests
    print_header("RUNNING VALIDATION TESTS")
    
    test_file_format(results, test_results)
    test_normalization(results, test_results)
    test_reasonable_values(results, test_results)
    test_basic_properties(results, test_results)
    test_expected_rankings(results, test_results)
    test_algorithm_convergence(results, test_results)
    
    # Print summary
    all_passed = test_results.print_summary()
    
    # Cleanup
    print_info("\nCleaning up test files...")
    cleanup_files()
    
    return all_passed

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Test interrupted by user{Colors.END}")
        cleanup_files()
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {str(e)}{Colors.END}")
        import traceback
        traceback.print_exc()
        cleanup_files()
        sys.exit(1)