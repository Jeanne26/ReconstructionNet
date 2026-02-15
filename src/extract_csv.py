import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

#extrait depuis les logs tensorboard
LOG_ROOT = "../logs"         
OUTPUT_DIR = "csv_exports" 

os.makedirs(OUTPUT_DIR,exist_ok=True)


def extract_scalars_from_event(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    scalar_tags = ea.Tags().get("scalars", [])
    all_data = []

    for tag in scalar_tags:
        events = ea.Scalars(tag)
        for ev in events:
            all_data.append({"tag": tag, "step": ev.step, "value": ev.value, "wall_time": ev.wall_time})
    df = pd.DataFrame(all_data)
    return df


def process_experiments(prefix):
    for i in range(10):
        folder_name = f"{prefix}_{i}"
        log_path = os.path.join(LOG_ROOT, folder_name)
        df = extract_scalars_from_event(log_path)

        output_path = os.path.join(OUTPUT_DIR, f"{folder_name}.csv")
        df.to_csv(output_path, index=False)


#calcul des moyennes et std pour les 10 runs de chaque dataset
OUTPUT_SUMMARY = "summary_results.csv"



def last_value(df, tag_name):
    tag_df = df[df["tag"] == tag_name]
    if tag_df.empty:
        return None
    return tag_df.sort_values("step")["value"].iloc[-1]


def process_prefix(prefix):
    train_accs = []
    test_accs = []
    test_aucs = []

    for i in range(10):
        file_path = os.path.join(OUTPUT_DIR, f"{prefix}_{i}.csv")
        df = pd.read_csv(file_path)
        train_acc = last_value(df, "RecNet/train_accuracy")
        test_acc = last_value(df, "RecNet/test_accuracy")
        test_auc = last_value(df, "RecNet/test_auc_epoch")
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        test_aucs.append(test_auc)

    results = {
        "dataset": prefix,
        "train_accuracy_mean": np.mean(train_accs),
        "train_accuracy_std": np.std(train_accs),
        "test_accuracy_mean": np.mean(test_accs),
        "test_accuracy_std": np.std(test_accs),
        "test_auc_mean": np.mean(test_aucs),
        "test_auc_std": np.std(test_aucs),
    }

    return results


if __name__ == "__main__":
    # process_experiments("mnist")

    # process_experiments("octmnist")
    process_experiments("diabetes")
    mnist_results = process_prefix("mnist")
    octmnist_results = process_prefix("octmnist")
    diabetes_results = process_prefix("diabetes")
    results_df = pd.DataFrame([mnist_results, octmnist_results, diabetes_results])
    print("here")
    results_df.to_csv(OUTPUT_SUMMARY, index=False)




