"""
Simple Windows-friendly test for Task 2 PageRank

This directly imports and runs the PageRank code without subprocess calls.
"""

import os
import sys
import shutil

def cleanup():
    """Clean up test files"""
    files = ['pagerank_input.txt', 'pagerank_output.txt']
    for f in files:
        if os.path.exists(f):
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)

def create_test_input():
    """Create test input file"""
    with open('pagerank_input.txt', 'w') as f:
        f.write("0 1\n")
        f.write("0 2\n")
        f.write("2 3\n")
    print("✓ Created test input file")

def read_output():
    """Read output file"""
    try:
        # Find the part file in output directory
        part_files = [f for f in os.listdir('pagerank_output.txt') if f.startswith('part-')]
        if not part_files:
            return None
        
        with open(os.path.join('pagerank_output.txt', part_files[0]), 'r') as f:
            results = {}
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    results[parts[0]] = float(parts[1])
            return results
    except Exception as e:
        print(f"✗ Error reading output: {e}")
        return None

def validate_results(results):
    """Validate the PageRank results"""
    print("\n=== VALIDATION RESULTS ===\n")
    
    passed = 0
    total = 0
    
    # Test 1: All vertices present
    total += 1
    if set(results.keys()) == {'0', '1', '2', '3'}:
        print("✓ Test 1 PASSED: All 4 vertices in output")
        passed += 1
    else:
        print(f"✗ Test 1 FAILED: Expected vertices 0,1,2,3, got {results.keys()}")
    
    # Test 2: All values positive
    total += 1
    if all(v > 0 for v in results.values()):
        print("✓ Test 2 PASSED: All PageRank values are positive")
        passed += 1
    else:
        print("✗ Test 2 FAILED: Some values are not positive")
    
    # Test 3: Normalization (sum ≈ 1.0)
    total += 1
    total_sum = sum(results.values())
    if abs(total_sum - 1.0) < 0.01:
        print(f"✓ Test 3 PASSED: Sum of ranks ≈ 1.0 (actual: {total_sum:.6f})")
        passed += 1
    else:
        print(f"✗ Test 3 FAILED: Sum should be ≈1.0, got {total_sum:.6f}")
    
    # Test 4: Values in valid range
    total += 1
    if all(0 < v < 1 for v in results.values()):
        print("✓ Test 4 PASSED: All values in range (0, 1)")
        passed += 1
    else:
        print("✗ Test 4 FAILED: Some values outside valid range")
    
    # Test 5: Values have differentiated
    total += 1
    values = list(results.values())
    if not all(abs(v - values[0]) < 0.0001 for v in values):
        print("✓ Test 5 PASSED: PageRank values have differentiated")
        passed += 1
    else:
        print("✗ Test 5 FAILED: All values are identical (algorithm may not have run)")
    
    print(f"\n{'='*50}")
    print(f"TOTAL: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! 🎉")
    else:
        print("⚠ Some tests failed")
    
    return passed == total

def main():
    print("="*50)
    print("TASK 2 PAGERANK - WINDOWS TEST")
    print("="*50)
    
    # Check if task2_pagerank.py exists
    if not os.path.exists('task2_pagerank.py'):
        print("✗ Error: task2_pagerank.py not found!")
        return False
    
    # Clean up
    print("\nCleaning up previous files...")
    cleanup()
    
    # Create test input
    print("Setting up test...")
    create_test_input()
    
    # Import and run the PageRank code
    print("\nRunning PageRank algorithm...")
    print("-"*50)
    
    try:
        # Add current directory to path so we can import
        sys.path.insert(0, os.getcwd())
        
        # Import the module
        import task2_pagerank
        
        # Run it
        task2_pagerank.main()
        
        print("-"*50)
        print("✓ PageRank execution completed")
        
    except Exception as e:
        print(f"✗ Error running PageRank: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Read and validate output
    print("\nReading output...")
    results = read_output()
    
    if results is None:
        print("✗ Could not read output file")
        return False
    
    print("✓ Output file read successfully")
    print("\nPageRank Results:")
    for vertex in sorted(results.keys()):
        print(f"  Vertex {vertex}: {results[vertex]:.6f}")
    
    # Validate
    success = validate_results(results)
    
    # Cleanup
    print("\nCleaning up test files...")
    cleanup()
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        cleanup()
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        cleanup()
        sys.exit(1)
        sys.exit(1)