"""
Clinical Diagnosis System - CLI Entry Point.

Simple command-line interface for testing the diagnosis system.
"""

import argparse
import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_tests():
    """Run all tests."""
    import pytest

    print("\n" + "="*60)
    print("Running Clinical Diagnosis System Tests")
    print("="*60)

    test_files = [
        "tests/test_rag.py",
        "tests/test_cag.py",
        "tests/test_preprocessing.py",
        "tests/test_conflict_resolver.py",
        "tests/test_diagnosis_flow.py"
    ]

    for test_file in test_files:
        if Path(test_file).exists():
            print(f"\n\nRunning {test_file}...")
            pytest.main([test_file, "-v"])


def init_vector_db():
    """Initialize vector database with sample documents."""
    print("\n" + "="*60)
    print("Initializing Vector Database")
    print("="*60)

    from scripts.knowledge_base.build_vector_db import initialize_vector_db

    vector_store = initialize_vector_db(reset=False, add_samples=True)

    print("\n✓ Vector database initialized successfully")


def generate_test_data():
    """Generate synthetic test data."""
    print("\n" + "="*60)
    print("Generating Test Data")
    print("="*60)

    from scripts.generate_test_data import main
    main()


def check_system():
    """Check system status."""
    print("\n" + "="*60)
    print("Clinical Diagnosis System Status")
    print("="*60)

    # Check components
    checks = {
        "Data Directory": Path("data").exists(),
        "Test Data": Path("data/test/microbiome_raw.csv").exists(),
        "Labeled Data": Path("data/labeled/annotations.json").exists(),
        "Knowledge Base": Path("data/knowledge_base").exists(),
        "Models Directory": Path("data/models").exists(),
    }

    print("\nData Status:")
    for check, status in checks.items():
        symbol = "✓" if status else "✗"
        color = "\033[92m" if status else "\033[91m"
        reset = "\033[0m"
        print(f"  {color}{symbol}{reset} {check}")

    # Check code modules
    print("\nSystem Modules:")
    modules = [
        "clinical.preprocessing",
        "clinical.experts",
        "clinical.collaboration",
        "clinical.decision",
        "mcp_server.clinical_diagnosis_server"
    ]

    for module in modules:
        try:
            __import__(module)
            print(f"  \033[92m✓\033[0m {module}")
        except Exception as e:
            print(f"  \033[91m✗\033[0m {module}: {e}")

    # Try importing RAG
    print("\nCollaboration Layer:")
    try:
        from clinical.collaboration.rag_system import RAGSystem
        rag = RAGSystem()
        stats = rag.get_statistics()
        print(f"  \033[92m✓\033[0m RAG System: {stats['total_documents']} documents")
    except Exception as e:
        print(f"  \033[91m✗\033[0m RAG System: {e}")

    try:
        from clinical.collaboration.cag_system import CAGSystem
        cag = CAGSystem()
        stats = cag.get_statistics()
        print(f"  \033[92m✓\033[0m CAG System: {stats['total_cases']} cases")
    except Exception as e:
        print(f"  \033[91m✗\033[0m CAG System: {e}")

    print("\n" + "="*60)


def train_models():
    """Train expert models."""
    print("\n" + "="*60)
    print("Training Expert Models")
    print("="*60)
    print("\nNOTE: This requires labeled training data.")
    print("Run 'python main_clinical.py generate-data' first if not done.\n")

    # Check if labeled data exists
    if not Path("data/labeled/annotations.json").exists():
        print("✗ Error: No labeled data found.")
        print("  Please run: python main_clinical.py generate-data")
        return

    # Import and run training
    from scripts.model_training.train_experts import main
    main()


def run_diagnosis_demo():
    """Run a demo diagnosis."""
    print("\n" + "="*60)
    print("Demo Diagnosis")
    print("="*60)

    # Check if test data exists
    if not Path("data/test/microbiome_raw.csv").exists():
        print("✗ Error: No test data found.")
        print("  Please run: python main_clinical.py generate-data")
        return

    # Run integration test
    from tests.test_diagnosis_flow import test_complete_workflow
    test_complete_workflow()


def show_menu():
    """Show interactive menu."""
    while True:
        print("\n" + "="*60)
        print("Clinical Diagnosis System - Main Menu")
        print("="*60)
        print("\n1. Check System Status")
        print("2. Generate Test Data")
        print("3. Initialize Vector Database")
        print("4. Train Expert Models")
        print("5. Run Tests")
        print("6. Run Demo Diagnosis")
        print("7. Exit")

        choice = input("\nEnter your choice (1-7): ").strip()

        if choice == "1":
            check_system()
        elif choice == "2":
            generate_test_data()
        elif choice == "3":
            init_vector_db()
        elif choice == "4":
            train_models()
        elif choice == "5":
            run_tests()
        elif choice == "6":
            run_diagnosis_demo()
        elif choice == "7":
            print("\nGoodbye!")
            break
        else:
            print("\n✗ Invalid choice. Please enter 1-7.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clinical Diagnosis System - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check system status
  python main_clinical.py status

  # Generate test data
  python main_clinical.py generate-data

  # Initialize vector database
  python main_clinical.py init-vectordb

  # Train expert models
  python main_clinical.py train

  # Run all tests
  python main_clinical.py test

  # Run demo diagnosis
  python main_clinical.py demo

  # Interactive menu
  python main_clinical.py
        """
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=["status", "generate-data", "init-vectordb", "train", "test", "demo"],
        help="Command to execute"
    )

    args = parser.parse_args()

    if args.command == "status":
        check_system()
    elif args.command == "generate-data":
        generate_test_data()
    elif args.command == "init-vectordb":
        init_vector_db()
    elif args.command == "train":
        train_models()
    elif args.command == "test":
        run_tests()
    elif args.command == "demo":
        run_diagnosis_demo()
    else:
        # No command - show interactive menu
        show_menu()


if __name__ == "__main__":
    main()
