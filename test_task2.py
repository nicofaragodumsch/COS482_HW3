#!/usr/bin/env python3
"""
Test Suite for Task 2: PageRank Implementation

This script tests that task2_pagerank.py correctly implements all requirements
from the assignment checklist.
"""

import os
import sys
import subprocess
import shutil
import time
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
        'test_pagerank_output.txt'
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

def run_spark_submit(script_name, input_file='pagerank_input.txt', output_file='pagerank_output.txt'):
    """Run spark-submit and return success status"""
    try:
        cmd = ['spark-submit', script_name, input_file, output_file]
        print_info(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        if result.returncode != 0:
            print_fail(f"spark-submit failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr[-500:]}")  # Last 500 chars
            return False
        
        return True
        
    except subprocess.TimeoutExpired:
        print_fail("spark-submit timed out after 120 seconds")
        return False
    except FileNotFoundError:
        print_fail("spark-submit command not found. Is Spark installed?")
        return False
    except Exception as e:
        print_fail(f"Error running spark-submit: {str(e)}")
        return False

def read_output(output_dir='pagerank_output.txt'):
    """Read the PageRank output"""
    try:
        # Output is in a directory, find the part file
        part_files = list(Path(output_dir).glob('part-*'))
        if not part_files:
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
    
    # Check that we have the expected vertices
    expected_vertices = {'0', '1', '2', '3'}
    actual_vertices = set(results.keys())
    
    if actual_vertices == expected_vertices:
        test_results.add_pass("All vertices present in output")
    else:
        test_results.add_fail("Output vertices", 
                             f"Expected {expected_vertices}, got {actual_vertices}")
    
    # Check that all values are floats
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
    
    # After normalization, sum should be close to 1.0
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
    
    # All values should be positive
    all_positive = all(v > 0 for v in results.values())
    if all_positive:
        test_results.add_pass("All PageRank values are positive")
    else:
        test_results.add_fail("PageRank values", "Some values are not positive")
    
    # All values should be less than 1 (after normalization)
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
    
    # Print actual values for inspection
    print_info("Actual PageRank values:")
    for vertex in sorted(results.keys()):
        print(f"    Vertex {vertex}: {results[vertex]:.6f}")
    
    # In the graph 0->1, 0->2, 2->3:
    # After adding edges for vertices with no outgoing edges (1 and 3):
    # - Vertex 1 gets links from 0 and 3
    # - Vertex 3 gets links from 2 and 1
    # - Vertex 0 gets links from 1 and 3
    # - Vertex 2 gets links from 0, 1, and 3
    
    # Vertex 2 should have relatively high PageRank (receives from 3 vertices)
    # The exact values depend on the algorithm, but we can check basic properties
    
    # Check that no vertex has zero PageRank
    no_zeros = all(v > 0.1 for v in results.values())
    if no_zeros:
        test_results.add_pass("No vertex has near-zero PageRank")
    else:
        test_results.add_fail("Rankings", "Some vertices have very low PageRank")
    
    # Check that ranks are reasonably distributed
    max_rank = max(results.values())
    min_rank = min(results.values())
    ratio = max_rank / min_rank if min_rank > 0 else float('inf')
    
    if ratio < 10:  # Reasonable distribution
        test_results.add_pass(f"PageRank distribution is reasonable (max/min ratio: {ratio:.2f})")
    else:
        test_results.add_fail("Rankings", f"PageRank too skewed (ratio: {ratio:.2f})")

def test_algorithm_convergence(results, test_results):
    """Test 5: Algorithm appears to have converged"""
    print_test("Algorithm convergence")
    
    if results is None:
        test_results.add_fail("Convergence", "No results to check")
        return
    
    # After 10 iterations, values should be stable
    # We can't directly test convergence without running multiple times,
    # but we can check that values are not at initialization (not all equal to 0.25)
    
    values = list(results.values())
    all_equal = all(abs(v - values[0]) < 0.0001 for v in values)
    
    if not all_equal:
        test_results.add_pass("PageRank values have differentiated (not stuck at initialization)")
    else:
        test_results.add_fail("Convergence", 
                             "All PageRank values are identical - algorithm may not have run")

def test_handles_no_outgoing_edges(test_results):
    """Test 6: Correctly handles nodes with no outgoing edges"""
    print_test("Handles vertices with no outgoing edges")
    
    # Create a special test case: vertex 4 has no outgoing edges
    test_input = 'test_no_outgoing.txt'
    test_output = 'test_no_outgoing_output.txt'
    
    try:
        # Create test graph: 0->1, 1->2, 2->0 (cycle), 3 has no edges
        with open(test_input, 'w') as f:
            f.write("0 1\n")
            f.write("1 2\n")
            f.write("2 0\n")
            # Vertex 3 is mentioned but has no outgoing edges
            f.write("2 3\n")
        
        print_info(f"Created test graph with vertex 3 having no outgoing edges")
        
        # Run PageRank
        success = run_spark_submit('task2_pagerank.py', test_input, test_output)
        
        if not success:
            test_results.add_fail("No outgoing edges", "Failed to run test case")
            return
        
        # Read results
        results = read_output(test_output)
        
        if results and len(results) == 4:
            # Vertex 3 should still have a PageRank value
            if '3' in results and results['3'] > 0:
                test_results.add_pass("Vertex with no outgoing edges has valid PageRank")
            else:
                test_results.add_fail("No outgoing edges", "Vertex 3 missing or has zero PageRank")
        else:
            test_results.add_fail("No outgoing edges", "Incorrect number of vertices in output")
    
    finally:
        # Cleanup
        if os.path.exists(test_input):
            os.remove(test_input)
        if os.path.exists(test_output):
            shutil.rmtree(test_output)

def test_basic_properties(results, test_results):
    """Test 7: Basic mathematical properties"""
    print_test("Basic mathematical properties")
    
    if results is None:
        test_results.add_fail("Basic properties", "No results to check")
        return
    
    # PageRank is a probability distribution
    total = sum(results.values())
    
    # Should sum to 1.0 (within floating point precision)
    if abs(total - 1.0) < 0.001:
        test_results.add_pass("Probability distribution (sum ≈ 1.0)")
    else:
        test_results.add_fail("Basic properties", 
                             f"Sum should be 1.0, got {total:.6f}")
    
    # All values should be in (0, 1)
    valid_range = all(0 < v < 1 for v in results.values())
    if valid_range:
        test_results.add_pass("All values in valid range (0, 1)")
    else:
        test_results.add_fail("Basic properties", "Some values outside (0, 1)")

def main():
    """Main test runner"""
    print_header("TASK 2 PAGERANK TEST SUITE")
    
    # Initialize test results tracker
    test_results = TestResults()
    
    # Check if task2_pagerank.py exists
    if not os.path.exists('task2_pagerank.py'):
        print_fail("task2_pagerank.py not found in current directory!")
        print_info("Please ensure the script is in the same directory as this test file.")
        sys.exit(1)
    
    # Clean up any existing files
    print_info("Cleaning up previous test files...")
    cleanup_files()
    
    # Create test input
    print_info("Setting up test environment...")
    create_test_input()
    
    # Run the PageRank implementation
    print_header("RUNNING PAGERANK IMPLEMENTATION")
    success = run_spark_submit('task2_pagerank.py')
    
    if not success:
        print_fail("Failed to run task2_pagerank.py successfully")
        print_info("Cannot continue with tests")
        test_results.add_fail("Execution", "Script failed to run")
        test_results.print_summary()
        return False
    
    print_pass("task2_pagerank.py executed successfully")
    
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
    test_handles_no_outgoing_edges(test_results)
    
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