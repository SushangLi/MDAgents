#!/usr/bin/env python3
"""
Run debate tests with LLM integration.

Usage:
    python scripts/run_debate_tests.py                  # Mock mode (default)
    python scripts/run_debate_tests.py --use-real-llm   # Real LLM mode
    python scripts/run_debate_tests.py --help           # Show help
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env.local
env_local = project_root / ".env.local"
if env_local.exists():
    load_dotenv(env_local)
    print(f"✓ Loaded environment from {env_local}")

import pytest


def check_api_keys():
    """Check which LLM API keys are available."""
    keys = {
        "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY")
    }

    available = [k for k, v in keys.items() if v]
    return available


def print_banner():
    """Print test banner."""
    print("\n" + "=" * 70)
    print(" " * 15 + "DEBATE SYSTEM TEST WITH LLM")
    print("=" * 70)


def print_config(use_real_llm: bool, available_keys: list):
    """Print test configuration."""
    print("\nTest Configuration:")
    print("-" * 70)
    print(f"  Mode: {'REAL LLM' if use_real_llm else 'MOCK LLM'}")

    if use_real_llm:
        if available_keys:
            print(f"  Available API Keys: {', '.join(available_keys)}")
            print("  ⚠️  WARNING: This will consume API credits!")
        else:
            print("  ⚠️  WARNING: No API keys found - will fall back to MOCK mode")
    else:
        print("  No API calls will be made (Mock mode)")

    print("-" * 70)


def save_test_stats(stats: dict, output_dir: Path):
    """Save test statistics to JSON file."""
    stats_file = output_dir / "debate_test_stats.json"
    stats_file.parent.mkdir(parents=True, exist_ok=True)

    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Test statistics saved to: {stats_file}")


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description="Run debate tests with LLM integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                  # Run in mock mode (no API calls)
  %(prog)s --use-real-llm   # Run with real LLM (requires API key)
        """
    )

    parser.add_argument(
        "--use-real-llm",
        action="store_true",
        help="Use real LLM instead of mock (requires API key)"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/test"),
        help="Output directory for reports (default: data/test)"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Print banner
    print_banner()

    # Check API keys
    available_keys = check_api_keys()
    print_config(args.use_real_llm, available_keys)

    # Prepare output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare pytest arguments
    pytest_args = [
        "tests/test_debate_with_llm.py",
        "-v" if args.verbose else "",
        "-s",  # Show print statements
        "--asyncio-mode=auto",
        "--tb=short",
        f"--html={output_dir}/debate_test_report.html",
        "--self-contained-html"
    ]

    # Remove empty strings
    pytest_args = [arg for arg in pytest_args if arg]

    # Set environment variable for real LLM mode
    if not args.use_real_llm:
        os.environ["FORCE_MOCK_LLM"] = "1"

    # Run tests
    print(f"\n{'=' * 70}")
    print("Running Tests...")
    print(f"{'=' * 70}\n")

    start_time = datetime.now()
    exit_code = pytest.main(pytest_args)
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    # Print results
    print(f"\n{'=' * 70}")
    print("Test Results")
    print(f"{'=' * 70}")
    print(f"  Exit Code: {exit_code}")
    print(f"  Duration: {elapsed:.2f} seconds")

    if exit_code == 0:
        print(f"  Status: ✅ ALL TESTS PASSED")
    else:
        print(f"  Status: ❌ SOME TESTS FAILED")

    print(f"{'=' * 70}")

    # Report locations
    print(f"\nGenerated Files:")
    print(f"  - HTML Report: {output_dir}/debate_test_report.html")
    print(f"  - Diagnostic Report: {output_dir}/test_debate_llm_report.md")

    # Save statistics
    stats = {
        "test_run": {
            "timestamp": start_time.isoformat(),
            "mode": "real_llm" if args.use_real_llm else "mock",
            "duration_seconds": elapsed,
            "exit_code": exit_code,
            "status": "passed" if exit_code == 0 else "failed",
            "available_api_keys": available_keys if args.use_real_llm else []
        }
    }

    save_test_stats(stats, output_dir)

    print(f"\n{'=' * 70}\n")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
