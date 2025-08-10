import sys
from src.run_experiment import run_experiment

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py configs/last_month.yaml")
        sys.exit(1)
    run_experiment(sys.argv[1])
