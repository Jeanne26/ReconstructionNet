from model import ReconstructionNet
import torch
import os
import numpy as np
import yaml
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import torch.nn.functional as F


CONFIG_PATH = '../configs/'


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


def load_model(config, model_path, device):
    is_image = config["is_image"]
    num_classes = config["num_classes"]
    data_folder = config["data_folder"]

    id_path = os.path.join(data_folder, config["data_name"])
    id_data = torch.load(id_path, weights_only=False)
    X_id = id_data['X_test']

    if is_image:
        input_dim = X_id.shape[1:]
    else:
        input_dim = X_id.shape[1]

    model = ReconstructionNet(input_dim, num_classes, is_image).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    return model



def compute_uncertainty(model, dataloader, device, num_classes):

    all_uncertainties = []
    all_predictions = []
    all_labels = []
    all_pred_probs = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits, r_list, wre = model(x)
            probs = F.softmax(-wre, dim=1)       # Eq. 6
            preds = torch.argmax(logits, dim=1)

            for i in range(x.size(0)):
                pred_class = preds[i].item()
                reconstruction = r_list[pred_class][i]
                original = x[i]

                # Eq.4 + Def.4: e_ij = (x̂ - x)², σ_dist = ||e||_1
                squared_error = (reconstruction - original) ** 2
                uncertainty = squared_error.sum().item()

                all_uncertainties.append(uncertainty)
                all_predictions.append(pred_class)
                all_pred_probs.append(probs[i].cpu().numpy())

                label = y[i]
                if label.dim() == 0:
                    all_labels.append(label.item())
                else:
                    all_labels.append(label.cpu().numpy())

    return (
        np.array(all_uncertainties),
        np.array(all_predictions),
        np.array(all_labels),
        np.array(all_pred_probs),
    )


def compute_auroc_ood(id_uncertainties, ood_uncertainties):
    labels = np.concatenate([
        np.zeros(len(id_uncertainties)),
        np.ones(len(ood_uncertainties)),
    ])
    scores = np.concatenate([id_uncertainties, ood_uncertainties])
    return roc_auc_score(labels, scores)


# Appendix A.3.1
def compute_correlation(uncertainties, pred_probs, true_labels, num_classes):

    one_hot = np.zeros((len(true_labels), num_classes))
    for i, label in enumerate(true_labels):
        one_hot[i, int(label)] = 1.0

    prediction_errors = np.mean(np.abs(one_hot - pred_probs), axis=1)

    if prediction_errors.std() == 0 or uncertainties.std() == 0:
        return float('nan')

    corr, _ = pearsonr(uncertainties, prediction_errors)
    return corr


#(Appendix A.3.2)
def compute_aurc(uncertainties, predictions, true_labels):

    n = len(uncertainties)
    errors = (predictions != true_labels).astype(float)

    sorted_indices = np.argsort(uncertainties)
    sorted_errors = errors[sorted_indices]

    cumulative_errors = np.cumsum(sorted_errors)
    coverages = np.arange(1, n + 1) / n
    risks = cumulative_errors / np.arange(1, n + 1)

    aurc = np.trapezoid(risks, coverages)
    return aurc


# (Appendix A.3.3)
def compute_sigma_risk(uncertainties, predictions, true_labels,
                       sigmas=(0.1, 0.2, 0.3, 0.4)):

    errors = (predictions != true_labels).astype(float)

    # Normaliser les incertitudes dans [0, 1]
    u_min, u_max = uncertainties.min(), uncertainties.max()
    if u_max - u_min < 1e-12:
        normalized = np.zeros_like(uncertainties)
    else:
        normalized = (uncertainties - u_min) / (u_max - u_min)

    results = {}
    for sigma in sigmas:
        mask = normalized <= sigma
        if mask.sum() == 0:
            results[sigma] = float('nan')
        else:
            results[sigma] = errors[mask].mean()

    return results


def evaluate_ood(config, model_path, ood_data_name, device=None,
                 batch_size=64, verbose=True):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = config["num_classes"]
    data_folder = config["data_folder"]

    id_path = os.path.join(data_folder, config["data_name"])
    id_data = torch.load(id_path, weights_only=False)
    X_id, y_id = id_data['X_test'], id_data['y_test']


    ood_path = os.path.join(data_folder, ood_data_name)
    ood_data = torch.load(ood_path, weights_only=False)
    X_ood = ood_data['X_test']
    y_ood = ood_data['y_test']

    assert X_id.shape[1:] == X_ood.shape[1:], \
        f"Dim mismatch: ID {X_id.shape[1:]} vs OOD {X_ood.shape[1:]}"


    model = load_model(config, model_path, device)

    if verbose:
        print(f"  Model       : {model_path}")
        print(f"  ID data     : {config['data_name']} ({X_id.shape[0]} samples)")
        print(f"  OOD data    : {ood_data_name} ({X_ood.shape[0]} samples)")


    id_loader = DataLoader(TensorDataset(X_id, y_id),
                           batch_size=batch_size, shuffle=False)
    id_unc, id_pred, id_lbl, id_probs = compute_uncertainty(
        model, id_loader, device, num_classes)


    ood_loader = DataLoader(TensorDataset(X_ood, y_ood),
                            batch_size=batch_size, shuffle=False)
    ood_unc, _, _, _ = compute_uncertainty(
        model, ood_loader, device, num_classes)

    accuracy = (id_pred == id_lbl).mean() * 100
    auroc = compute_auroc_ood(id_unc, ood_unc)
    correlation = compute_correlation(id_unc, id_probs, id_lbl, num_classes)
    aurc = compute_aurc(id_unc, id_pred, id_lbl)
    sigma_risks = compute_sigma_risk(id_unc, id_pred, id_lbl)

    results = {
        'accuracy': accuracy,
        'auroc_ood': auroc,
        'correlation': correlation,
        'aurc': aurc,
    }
    for s, v in sigma_risks.items():
        results[f'sigma_risk_{s}'] = v

    if verbose:
        print(f"  Accuracy    : {accuracy:.2f}%")
        print(f"  AUROC OOD   : {auroc:.4f}")
        print(f"  Corrélation : {correlation:.4f}")
        print(f"  AURC        : {aurc:.4f}")
        for s, v in sigma_risks.items():
            print(f"  σ-Risk({s}) : {v:.4f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="OOD Evaluation for ReconstructionNet")
    parser.add_argument('--config', type=str, required=True,
                        help='Config YAML (ex: config_octmnist.yaml)')
    parser.add_argument('--model', type=str, required=True,
                        help='Chemin vers les poids du modèle')
    parser.add_argument('--ood_data', type=str, required=True,
                        help='Nom du fichier OOD (ex: chestmnist_processed.pt)')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    results = evaluate_ood(
        config, args.model, args.ood_data,
        device=device, batch_size=args.batch_size, verbose=True,
    )


