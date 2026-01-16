#!/usr/bin/env python3
"""Test runner script for PKoffee project.

This script provides convenient commands for running tests with various options.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and print status."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, check=False)
    
    if result.returncode == 0:
        print(f"\n✓ {description} PASSED")
    else:
        print(f"\n✗ {description} FAILED")
    
    return result.returncode


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description="PKoffee Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python run_tests.py
  
  # Run tests with coverage
  python run_tests.py --coverage
  
  # Run specific test file
  python run_tests.py --file test_metrics.py
  
  # Run tests matching pattern
  python run_tests.py --pattern "test_compute"
  
  # Run in verbose mode
  python run_tests.py --verbose
  
  # Run only fast tests (exclude slow ones)
  python run_tests.py --fast
        """
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Run tests with coverage report"
    )
    
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Run specific test file"
    )
    
    parser.add_argument(
        "--pattern", "-p",
        type=str,
        help="Run tests matching pattern (e.g., 'test_compute')"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip slow tests"
    )
    
    parser.add_argument(
        "--failfast", "-x",
        action="store_true",
        help="Stop on first failure"
    )
    
    args = parser.parse_args()
    
    # Build pytest command
    cmd = ["pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-vv")
    else:
        cmd.append("-v")
    
    # Add coverage
    if args.coverage:
        cmd.extend(["--cov=pkoffee", "--cov-report=html", "--cov-report=term"])
    
    # Add specific file
    if args.file:
        test_path = Path("tests") / args.file
        if not test_path.exists():
            test_path = Path(args.file)
        cmd.append(str(test_path))
    else:
        cmd.append("tests")
    
    # Add pattern matching
    if args.pattern:
        cmd.extend(["-k", args.pattern])
    
    # Skip slow tests
    if args.fast:
        cmd.extend(["-m", "not slow"])
    
    # Fail fast
    if args.failfast:
        cmd.append("-x")
    
    # Run tests
    return_code = run_command(cmd, "PKoffee Tests")
    
    # Print coverage location if coverage was run
    if args.coverage and return_code == 0:
        print("\n" + "="*70)
        print("📊 Coverage report generated!")
        print("HTML report: htmlcov/index.html")
        print("="*70)
    
    sys.exit(return_code)


if __name__ == "__main__":
    main()
