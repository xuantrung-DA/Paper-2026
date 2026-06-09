from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]

COMMANDS = [
    ["scripts/collect_results.py"],
    ["scripts/make_tables.py"],
    ["scripts/plot_training_curves.py"],
    ["scripts/plot_utility_bitrate.py"],
    ["scripts/plot_ablation.py"],
    ["scripts/plot_confusion_matrices.py"],
]


def main():
    for command in COMMANDS:
        print("=" * 80)
        print("[RUN]", " ".join(command))
        subprocess.run([sys.executable, str(ROOT / command[0]), *command[1:]], cwd=ROOT, check=True)
    print("=" * 80)
    print("[DONE] artifacts generated")


if __name__ == "__main__":
    main()
