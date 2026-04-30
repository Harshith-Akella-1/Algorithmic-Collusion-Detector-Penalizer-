"""
run.py -- End-to-end pipeline orchestrator.

Chains: eval_cnn → train_mappo → infer_rl → viz
Assumes dataset/ and detectors/cnn_best.pt already exist.

Usage:
    python run.py                    # full pipeline
    python run.py --skip-mappo       # skip MAPPO training
    python run.py --skip-viz         # skip visualization
    python run.py --mappo-iters 50   # short MAPPO training
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()


def banner(title):
    print()
    print('=' * 65)
    print(f'  STEP: {title}')
    print('=' * 65)
    print()


def run_step(title, cmd, cwd=None, skip=False, check_output=None):
    """Run a pipeline step. Returns True on success."""
    banner(title)

    if skip:
        print(f'  [SKIPPED]')
        return True

    if check_output and Path(check_output).exists():
        print(f'  Output already exists: {check_output}')
        print(f'  (delete to re-run)')
        return True

    cwd = cwd or str(PROJECT_ROOT)
    t0 = time.time()
    print(f'  CMD: {" ".join(cmd)}')
    print(f'  CWD: {cwd}')
    print()

    try:
        result = subprocess.run(
            cmd, cwd=cwd,
            stdout=sys.stdout, stderr=sys.stderr,
        )
        elapsed = time.time() - t0
        print(f'\n  [{title}] finished in {elapsed:.1f}s '
              f'(exit code: {result.returncode})')
        return result.returncode == 0
    except Exception as e:
        print(f'\n  [{title}] FAILED: {e}')
        return False


def check_prerequisites():
    """Verify that dataset and model exist."""
    labels = PROJECT_ROOT / 'dataset' / 'labels.parquet'
    seq = PROJECT_ROOT / 'dataset' / 'sequences.npy'
    idx = PROJECT_ROOT / 'dataset' / 'seq_index.parquet'
    model = PROJECT_ROOT / 'detectors' / 'cnn_best.pt'

    ok = True
    for path, desc in [(labels, 'Dataset labels'),
                       (seq, 'Sequences (sequences.npy)'),
                       (idx, 'Sequence index'),
                       (model, 'CNN model checkpoint')]:
        if not path.exists():
            print(f'  [MISSING] {desc} not found: {path}')
            ok = False
        else:
            print(f'  [OK] {desc}: {path}')

    return ok


def main():
    parser = argparse.ArgumentParser(
        description='Run the full collusion detection pipeline')
    parser.add_argument('--skip-mappo', action='store_true',
                        help='Skip MAPPO training')
    parser.add_argument('--skip-viz', action='store_true',
                        help='Skip visualization generation')
    parser.add_argument('--mappo-iters', type=int, default=100,
                        help='Number of MAPPO training iterations (default: 100)')
    args = parser.parse_args()

    py = sys.executable

    print()
    print('╔' + '═' * 63 + '╗')
    print('║  ALGORITHMIC COLLUSION DETECTOR — FULL PIPELINE               ║')
    print('╚' + '═' * 63 + '╝')
    print()

    # Check prerequisites
    banner('Prerequisites Check')
    if not check_prerequisites():
        print('\n  ABORT: Missing prerequisites. Generate the dataset and train CNN first.')
        print('  Run:')
        print('    cd Simulator && python generate.py')
        print('    cd data_prep && python features.py')
        print('    cd data_prep && python prepare_sequences.py')
        print('    cd detectors && python train_cnn.py')
        sys.exit(1)
    print('\n  All prerequisites found.')

    t_start = time.time()
    steps_ok = 0
    steps_total = 0

    # Step 1: Evaluate CNN (+ save confusion matrix PNG)
    steps_total += 1
    ok = run_step(
        'CNN Evaluation + Confusion Matrix',
        [py, 'eval_cnn.py'],
        cwd=str(PROJECT_ROOT / 'detectors'),
    )
    if ok:
        steps_ok += 1

    # Step 2: Train MAPPO
    steps_total += 1
    ok = run_step(
        'MAPPO Training (IPPO)',
        [py, 'train_mappo.py', '--iters', str(args.mappo_iters)],
        cwd=str(PROJECT_ROOT / 'rl_bots'),
        skip=args.skip_mappo,
    )
    if ok:
        steps_ok += 1

    # Step 3: Infer RL episodes (PPO)
    ppo_checkpoint = PROJECT_ROOT / 'rl_bots' / 'ppo_best.pt'
    steps_total += 1
    ok = run_step(
        'RL Inference — PPO Agent',
        [py, 'infer_rl_episodes.py', '--agent', 'ppo'],
        skip=not ppo_checkpoint.exists(),
    )
    if ok:
        steps_ok += 1

    # Step 4: Infer RL episodes (MAPPO)
    mappo_dir = PROJECT_ROOT / 'rl_bots' / 'mappo_episodes'
    steps_total += 1
    ok = run_step(
        'RL Inference — MAPPO Agents',
        [py, 'infer_rl_episodes.py', '--agent', 'mappo'],
        skip=args.skip_mappo or not mappo_dir.exists(),
    )
    if ok:
        steps_ok += 1

    # Step 5: Visualization
    steps_total += 1
    ok = run_step(
        'Visualization',
        [py, 'viz.py'],
        skip=args.skip_viz,
    )
    if ok:
        steps_ok += 1

    # Summary
    total_time = (time.time() - t_start) / 60
    print()
    print('╔' + '═' * 63 + '╗')
    print('║  PIPELINE COMPLETE                                            ║')
    print('╚' + '═' * 63 + '╝')
    print(f'  Steps passed: {steps_ok}/{steps_total}')
    print(f'  Total time:   {total_time:.1f} min')
    print()

    # List generated artifacts
    artifacts = [
        ('cnn_confusion_matrix.png', 'detectors'),
        ('collusion_evidence.png', '.'),
        ('mappo_training_curves.png', 'rl_bots'),
        ('mappo_history.json', 'rl_bots'),
        ('mappo_a_best.pt', 'rl_bots'),
        ('mappo_b_best.pt', 'rl_bots'),
    ]
    print('  Generated artifacts:')
    for name, subdir in artifacts:
        path = PROJECT_ROOT / subdir / name
        status = '[OK]' if path.exists() else '    '
        print(f'    {status} {subdir}/{name}')
    print()


if __name__ == '__main__':
    main()
