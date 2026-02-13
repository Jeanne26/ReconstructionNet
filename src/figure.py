import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse

#exempl d'appel :python3 figure.py --log mnist_3 --figure 1

#list figures: 1. confusion matrix

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, required=True)
parser.add_argument('--figure', type=str, required=True)

args = parser.parse_args()

tb_log_folder = args.log
tb_log_path = os.path.join("../logs", tb_log_folder)
tb_log_file = os.path.join(tb_log_path, os.listdir(tb_log_path)[0])
figure = args.figure

ea = EventAccumulator(tb_log_file)
ea.Reload()

def confusion_matrix():
    if "mnist" in tb_log_file:
        num_classes = 10
        title = "MNIST"
    elif "diabetes" in tb_log_file:
        num_classes = 2
        title = "Diabetes"

    
    matrix_event = ea.Tensors('RecNet/confusion_matrix/text_summary')[0]
    matrix_str = matrix_event.tensor_proto.string_val[0].decode()
    clean_str = (matrix_str.replace('[', '').replace(']', '').strip())

    matrix = np.array([
        [int(num) for num in line.split()]
        for line in clean_str.splitlines()
    ])

    plt.figure(figsize=(10, 7))
    sns.heatmap(matrix, cmap='Blues',annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {title}')
    plt.xticks(np.arange(num_classes)+0.5, labels=[f'Class {i}' for i in range(num_classes)], rotation=45)
    plt.yticks(np.arange(num_classes)+0.5, labels=[f'Class {i}' for i in range(num_classes)], rotation=0)
    plt.show()
    plt.savefig(f'../figures/confusion_matrix_{title}.png')
    

if __name__ == "__main__":
    
    if figure == "1":
        confusion_matrix()