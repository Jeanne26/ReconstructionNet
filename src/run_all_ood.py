import os
import sys
import glob
import pandas as pd
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ood_eval import load_config, evaluate_ood


COMBINATIONS = [
    {
        'id_name': 'MNIST',
        'config': 'config_mnist.yaml',
        'model_pattern': '../models/model_minst_*.pth',
        'ood_datasets': [
            ('ChestMNIST', 'chestmnist_processed.pt'),
            ('TissueMNIST', 'tissuemnist_processed.pt'),
        ],
    },
    {
        'id_name': 'OCTMNIST',
        'config': 'config_octmnist.yaml',
        'model_pattern': '../models/model_octminst_*.pth',
        'ood_datasets': [
            ('ChestMNIST', 'chestmnist_processed.pt'),
            ('TissueMNIST', 'tissuemnist_processed.pt'),
        ],
    },
]


def find_models(pattern):
    models = sorted(glob.glob(pattern))
    if not models:
        print(f"Aucun modèle trouvé pour le pattern : {pattern}")
    return models


def extract_seed(model_path):
    base = os.path.basename(model_path)      # model_minst_3.pth
    name = os.path.splitext(base)[0]  # model_minst_3
    seed = name.split('_')[-1]# 3
    return int(seed)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch OOD evaluation")
    parser.add_argument('--output', type=str,
                        default='../results/ood_results.csv',
                        help='Chemin du fichier CSV de sortie')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Output : {args.output}")
    print()

    all_rows = []

    for combo in COMBINATIONS:
        id_name = combo['id_name']
        config = load_config(combo['config'])
        models = find_models(combo['model_pattern'])

        for ood_label, ood_file in combo['ood_datasets']:
            print("=" * 70)
            print(f"  {id_name}  →  {ood_label}")
            print("=" * 70)

            for model_path in models:
                seed = extract_seed(model_path)
                print(f"\n  --- Seed {seed} ({os.path.basename(model_path)}) ---")

                try:
                    results = evaluate_ood(
                        config, model_path, ood_file,
                        device=device,
                        batch_size=args.batch_size,
                        verbose=True,
                    )

                    row = {
                        'id_dataset': id_name,
                        'ood_dataset': ood_label,
                        'seed': seed,
                        'model': os.path.basename(model_path),
                        **results,
                    }
                    all_rows.append(row)

                except Exception as e:
                    print(f"  ✗ ERREUR : {e}")
                    all_rows.append({
                        'id_dataset': id_name,
                        'ood_dataset': ood_label,
                        'seed': seed,
                        'model': os.path.basename(model_path),
                        'error': str(e),
                    })

            print()


    df = pd.DataFrame(all_rows)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(args.output, index=False)


    metric_cols = [c for c in df.columns
                   if c not in ('id_dataset', 'ood_dataset', 'seed', 'model', 'error')]



    summary_rows = []
    for (id_ds, ood_ds), group in df.groupby(['id_dataset', 'ood_dataset']):
        print(f"\n  {id_ds} → {ood_ds}  ({len(group)} seeds)")
        row = {'id_dataset': id_ds, 'ood_dataset': ood_ds, 'n_seeds': len(group)}
        for col in metric_cols:
            vals = pd.to_numeric(group[col], errors='coerce').dropna()
            if len(vals) > 0:
                mean, std = vals.mean(), vals.std()
                print(f"    {col:20s} : {mean:.4f} ± {std:.4f}")
                row[f'{col}_mean'] = mean
                row[f'{col}_std'] = std
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = args.output.replace('.csv', '_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Résumé sauvegardé dans : {summary_path}")


if __name__ == "__main__":
    main()
