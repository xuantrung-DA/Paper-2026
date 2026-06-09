from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


DEFAULT_CONFIGS = [
    "configs/baseline_mbv3.yaml",
    "configs/aqb_z64_b8.yaml",
]


def ensure_processed_csvs():
    required = [
        ROOT / "data/processed/train.csv",
        ROOT / "data/processed/val.csv",
        ROOT / "data/processed/test.csv",
    ]
    if all(path.exists() for path in required):
        return

    print("[INDEX] processed CSVs missing; running build_index.py --full")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_index.py"), "--full"],
        cwd=ROOT,
        check=True,
    )


def run_config(config_path: str, resume: bool = False):
    config = ROOT / config_path
    if not config.exists():
        raise FileNotFoundError(config)

    print("=" * 80)
    print(f"[RUN] {config_path}")
    command = [sys.executable, str(ROOT / "src/train.py"), "--config", str(config)]
    if resume:
        command.append("--resume")
    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def test_config(config_path: str):
    config = ROOT / config_path
    print("-" * 80)
    print(f"[TEST] {config_path}")
    subprocess.run(
        [sys.executable, str(ROOT / "src/test.py"), "--config", str(config)],
        cwd=ROOT,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Configs to run in order.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Regenerate data/processed CSVs before training.",
    )
    parser.add_argument(
        "--skip-artifacts",
        action="store_true",
        help="Do not generate tables and figures after training.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Do not run test evaluation after each training run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each training run from last.pt, or best.pt if last.pt is missing.",
    )
    args = parser.parse_args()

    if args.rebuild_index:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/build_index.py"), "--full"],
            cwd=ROOT,
            check=True,
        )
    else:
        ensure_processed_csvs()

    for config in args.configs:
        run_config(config, resume=args.resume)
        if not args.skip_test:
            test_config(config)

    if not args.skip_artifacts:
        print("=" * 80)
        print("[ARTIFACTS] generating tables and figures")
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/make_artifacts.py")],
            cwd=ROOT,
            check=True,
        )

    print("=" * 80)
    print("[DONE] pipeline finished")


if __name__ == "__main__":
    main()
