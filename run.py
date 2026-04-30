"""
run.py -- End-to-end pipeline orchestrator.

Chains: eval_cnn → train_mappo → infer_rl → viz

Per-step prereq checks: only the CNN checkpoint is mandatory. eval_cnn
auto-skips if the dataset/ files aren't present, so the rest of the
pipeline (MAPPO, RL inference, viz) still runs without them.

Usage:
    python run.py                    # full pipeline
    python run.py --skip-eval        # skip CNN evaluation
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


def run_step(title, cmd, cwd=None, skip=False, skip_reason='', check_output=None):
    """Run a pipeline step. Returns True on success."""
    banner(title)

    if skip:
        msg = '  [SKIPPED]'
        if skip_reason:
            msg += f' -- {skip_reason}'
        print(msg)
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
    """Per-step prereq check. Returns dict of available components.

    Only the CNN checkpoint is strictly required. Dataset files are only
    needed for eval_cnn.py; if they're missing, that step auto-skips.
    """
    labels = PROJECT_ROOT / 'dataset' / 'labels.parquet'
    seq = PROJECT_ROOT / 'dataset' / 'sequences.npy'
    idx = PROJECT_ROOT / 'dataset' / 'seq_index.parquet'
    model = PROJECT_ROOT / 'detectors' / 'cnn_best.pt'

    status = {
        'cnn_model': model.exists(),
        'dataset_labels': labels.exists(),
        'dataset_sequences': seq.exists(),
        'dataset_index': idx.exists(),
    }
    status['dataset_full'] = (status['dataset_labels']
                              and status['dataset_sequences']
                              and status['dataset_index'])

    def mark(ok):
        return '[OK]    ' if ok else '[MISSING]'

    print(f'  {mark(status["cnn_model"])}      CNN checkpoint   {model}')
    print(f'  {mark(status["dataset_labels"])}      Dataset labels   {labels}')
    print(f'  {mark(status["dataset_sequences"])}      Sequences        {seq}')
    print(f'  {mark(status["dataset_index"])}      Sequence index   {idx}')

    return status


def main():
    parser = argparse.ArgumentParser(
        description='Run the full collusion detection pipeline')
    parser.add_argument('--skip-eval', action='store_true',
                        help='Skip CNN evaluation (auto-skipped if dataset missing)')
    parser.add_argument('--skip-mappo', action='store_true',
                        help='Skip MAPPO training')
    parser.add_argument('--skip-viz', action='store_true',
                        help='Skip visualization generation')
    parser.add_argument('--mappo-iters', type=int, default=100,
                        help='Number of MAPPO training iterations (default: 100)')
    args = parser.parse_args()

    py = sys.executable

    print()
    print('+' + '=' * 63 + '+')
    print('|  ALGORITHMIC COLLUSION DETECTOR -- FULL PIPELINE               |')
    print('+' + '=' * 63 + '+')
    print()

    # Per-step prereq check
    banner('Prerequisites Check')
    status = check_prerequisites()

    # Mandatory: CNN model. Without it, every downstream step fails.
    if not status['cnn_model']:
        print('\n  ABORT: CNN checkpoint missing. Train the CNN first:')
        print('    cd Simulator && python generate.py')
        print('    cd data_prep && python features.py')
        print('    cd data_prep && python prepare_sequences.py')
        print('    cd detectors && python train_cnn.py')
        sys.exit(1)

    # eval_cnn auto-skips when dataset is missing
    skip_eval = args.skip_eval or not status['dataset_full']
    eval_reason = ''
    if args.skip_eval:
        eval_reason = '--skip-eval flag set'
    elif not status['dataset_full']:
        eval_reason = 'dataset/ files missing (regenerate to enable)'

    print('\n  CNN checkpoint OK -- proceeding.')
    if skip_eval:
        print(f'  eval_cnn will be skipped: {eval_reason}')

    t_start = time.time()
    steps_ok = 0
    steps_total = 0

    # ------------------------------------------------------------------
    # Step 1: Evaluate CNN (+ save confusion matrix PNG)
    # ------------------------------------------------------------------
    steps_total += 1
    ok = run_step(
        'CNN Evaluation + Confusion Matrix',
        [py, str(PROJECT_ROOT / 'detectors' / 'eval_cnn.py')],
        cwd=str(PROJECT_ROOT),
        skip=skip_eval,
        skip_reason=eval_reason,
    )
    if ok:
        steps_ok += 1

    # ------------------------------------------------------------------
    # Step 2: Train MAPPO (IPPO)
    # ------------------------------------------------------------------
    steps_total += 1
    ok = run_step(
        'MAPPO Training (IPPO)',
        [py, 'train_mappo.py', '--iters', str(args.mappo_iters)],
        cwd=str(PROJECT_ROOT / 'rl_bots'),
        skip=args.skip_mappo,
        skip_reason='--skip-mappo flag set' if args.skip_mappo else '',
    )
    if ok:
        steps_ok += 1

    # ------------------------------------------------------------------
    # Step 3: Infer RL episodes (single-agent PPO)
    # ------------------------------------------------------------------
    ppo_checkpoint = PROJECT_ROOT / 'rl_bots' / 'ppo_best.pt'
    skip_ppo_inf = not ppo_checkpoint.exists()
    steps_total += 1
    ok = run_step(
        'RL Inference -- PPO Agent',
        [py, 'infer_rl_episodes.py', '--agent', 'ppo'],
        skip=skip_ppo_inf,
        skip_reason='ppo_best.pt not found (train_rl.py to generate)'
                    if skip_ppo_inf else '',
    )
    if ok:
        steps_ok += 1

    # ------------------------------------------------------------------
    # Step 4: Infer RL episodes (MAPPO/IPPO pair)
    # ------------------------------------------------------------------
    mappo_dir = PROJECT_ROOT / 'rl_bots' / 'mappo_episodes'
    skip_mappo_inf = args.skip_mappo or not mappo_dir.exists()
    if args.skip_mappo:
        mappo_inf_reason = '--skip-mappo flag set'
    elif not mappo_dir.exists():
        mappo_inf_reason = 'mappo_episodes/ not found (run MAPPO training first)'
    else:
        mappo_inf_reason = ''
    steps_total += 1
    ok = run_step(
        'RL Inference -- MAPPO Agents',
        [py, 'infer_rl_episodes.py', '--agent', 'mappo'],
        skip=skip_mappo_inf,
        skip_reason=mappo_inf_reason,
    )
    if ok:
        steps_ok += 1

    # ------------------------------------------------------------------
    # Step 5: Visualization
    # ------------------------------------------------------------------
    steps_total += 1
    ok = run_step(
        'Visualization',
        [py, 'viz.py'],
        skip=args.skip_viz,
        skip_reason='--skip-viz flag set' if args.skip_viz else '',
    )
    if ok:
        steps_ok += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_time = (time.time() - t_start) / 60
    print()
    print('+' + '=' * 63 + '+')
    print('|  PIPELINE COMPLETE                                            |')
    print('+' + '=' * 63 + '+')
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
        marker = '[OK]' if path.exists() else '    '
        print(f'    {marker} {subdir}/{name}')
    print()


if __name__ == '__main__':
    main()