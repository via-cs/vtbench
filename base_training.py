import os
import numpy as np
import pandas as pd
import re
from data_utils import read_ucr, normalize_data
from OS_CNN_easy_use import OS_CNN_easy_use

Result_log_folder = 'Results_of_OS_CNN'
os.makedirs(Result_log_folder, exist_ok=True)

excel_file_path = os.path.join(Result_log_folder, "OSCNN_results.xlsx")

def get_unique_filename(base_path):
    if not os.path.exists(base_path):
        return base_path
    counter = 1
    while os.path.exists(f"{base_path.replace('.xlsx', f'_{counter}.xlsx')}"):
        counter += 1
    return base_path.replace('.xlsx', f'_{counter}.xlsx')

excel_file_path = get_unique_filename(excel_file_path)


def write_epoch_results_to_excel(file_path, epoch_results):
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        df = pd.DataFrame()

    results_df = pd.DataFrame(epoch_results)
    df = pd.concat([df, results_df], ignore_index=True)

    with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
        df.to_excel(writer, index=False)


def parse_log_file(log_file, dataset_name):
    results = []
    with open(log_file, 'r') as f:
        lines = f.readlines()

    epoch_num = None
    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith("epoch"):
            match_epoch = re.search(r"epoch\s*=\s*(\d+)", line)
            if match_epoch:
                epoch_num = int(match_epoch.group(1))
        elif "train_acc=" in line and "test_acc=" in line:
            match_train = re.search(r"train_acc=\s*([\d\.]+)", line)
            match_test = re.search(r"test_acc=\s*([\d\.]+)", line)
            if match_train and match_test and epoch_num is not None:
                results.append({
                    "Dataset": dataset_name,
                    "Epoch": epoch_num,
                    "Train Accuracy": float(match_train.group(1)),
                    "Test Accuracy": float(match_test.group(1))
                })
    return results


# List of datasets to run
dataset_list = ["Yoga", "Strawberry"]  # Add more as needed

for dataset_name in dataset_list:
    print(f"\n=== Running on dataset: {dataset_name} ===")

    train_file = f"../../data/{dataset_name}/{dataset_name}_TRAIN.ts"
    test_file = f"../../data/{dataset_name}/{dataset_name}_TEST.ts"

    x_train, y_train = read_ucr(train_file)
    x_test, y_test = read_ucr(test_file)
    x_train, x_test = normalize_data(x_train, x_test)

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    model = OS_CNN_easy_use(
        Result_log_folder=Result_log_folder,
        dataset_name=dataset_name,
        device="cuda:0",
        max_epoch=500,
        paramenter_number_of_layer_list=[8 * 128, 5 * 128 * 256 + 2 * 256 * 128],
    )

    model.fit(x_train, y_train, x_test, y_test)

    log_file = f"Results_of_OS_CNN{dataset_name}/{dataset_name}_.txt"
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        continue

    epoch_results = parse_log_file(log_file, dataset_name)
    if epoch_results:
        write_epoch_results_to_excel(excel_file_path, epoch_results)
        print(f"✅ Epoch results written to: {excel_file_path}")
    else:
        print(f"No epoch results found in log for {dataset_name}")
