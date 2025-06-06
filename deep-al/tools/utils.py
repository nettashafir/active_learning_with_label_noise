import os
import random
import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed)  # Python random module.
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU.

def get_budget(noise_ratio, samples_per_class, num_classes):
  budget = int(np.ceil((samples_per_class * num_classes) / (1-noise_ratio)))
  return budget

def save_stats(path, test_accuracies, clean_true_positives, clean_false_positives, clean_false_negatives, clean_true_negatives):
    np.save(os.path.join(path, 'test_accuracies.npy'), test_accuracies)
    np.save(os.path.join(path, 'clean_true_positives.npy'), clean_true_positives)
    np.save(os.path.join(path, 'clean_true_negatives.npy'), clean_true_negatives)
    np.save(os.path.join(path, 'clean_false_positives.npy'), clean_false_positives)
    np.save(os.path.join(path, 'clean_false_negatives.npy'), clean_false_negatives)

def calc_metrics(clean_tp, clean_fp, clean_tn, clean_fn):
    accuracy = (clean_tp + clean_tn) / (clean_tp + clean_fp + clean_tn + clean_fn)

    clean_precision = clean_tp / (clean_tp + clean_fp)
    clean_recall = clean_tp / (clean_tp + clean_fn)
    clean_f1 = 2 * (clean_precision * clean_recall) / (clean_precision + clean_recall)

    noise_precision = clean_tn / (clean_tn + clean_fn)
    noise_recall = clean_tn / (clean_tn + clean_fp)
    noise_f1 = 2 * (noise_precision * noise_recall) / (noise_precision + noise_recall)

    predicted_noise = (clean_tn + clean_fn) / (clean_tn + clean_fn + clean_tp + clean_fp)
    true_noise = (clean_tn + clean_fp) / (clean_tn + clean_fn + clean_tp + clean_fp)

    return accuracy, clean_precision, clean_recall, clean_f1, noise_precision, noise_recall, noise_f1, true_noise, predicted_noise


def print_table(data, column_headers=None, row_headers=None, padding=2, logger=None):
    """
    Print a formatted table from a list of lists with optional column and row headers.

    Args:
        data: List of lists containing the table data
        column_headers: List of column headers
        row_headers: List of row headers
        padding: Number of spaces for padding between columns (default=2)
    """

    # Ensure all data is converted to strings
    str_data = [[str(cell) for cell in row] for row in data]

    # Set default headers if not provided
    if column_headers is None:
        column_headers = [f"Column {i +1}" for i in range(len(data[0]))]
    if row_headers is None:
        row_headers = [f"Row {i +1}" for i in range(len(data))]

    # Convert headers to strings
    column_headers = [str(h) for h in column_headers]
    row_headers = [str(h) for h in row_headers]

    # Calculate row header width
    row_header_width = max(len(h) for h in row_headers)

    # Calculate column widths
    col_widths = []
    for i in range(len(column_headers)):
        column = [row[i] for row in str_data]
        col_widths.append(max(len(str(x)) for x in column + [column_headers[i]]))

    # Create format strings
    row_header_format = "{:<" + str(row_header_width) + "}" + " " * padding
    col_format = (" " * padding).join("{:<" + str(width) + "}" for width in col_widths)

    # Print column headers
    print(row_header_format.format(""), end="")
    print(col_format.format(*column_headers))
    if logger is not None:
        logger.info(row_header_format.format("") + col_format.format(*column_headers))

    # Print separator line
    total_width = row_header_width + padding + sum(col_widths) + padding * len(col_widths)
    print("-" * total_width)
    if logger is not None:
        logger.info("-" * total_width)

    # Print data rows with row headers
    for header, row in zip(row_headers, str_data):
        print(row_header_format.format(header), end="")
        print(col_format.format(*row))
        if logger is not None:
            logger.info(row_header_format.format(header) + col_format.format(*row))
    print("\n")
    if logger is not None:
        logger.info("\n")
