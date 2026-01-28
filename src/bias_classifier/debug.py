"""Quick debug script to inspect the dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bias_classifier.dataset import get_classification_dataset, transform_to_classification
from bias_classifier.utils import load_tokenizer
from dataset.constants import DATASET_SLICE_BY_SPLIT_TYPE
from dataset.enums import DatasetSplit
from dataset.preprocess import get_preprocessed_dataset_slice


def debug_dataset():
    # Load raw data first
    print("=== Raw WNC Data (first 5 pairs) ===")
    slice_range = DATASET_SLICE_BY_SPLIT_TYPE[DatasetSplit.TRAIN]
    raw_dataset = get_preprocessed_dataset_slice(slice_range)

    for i in range(5):
        print(f"\nPair {i}:")
        print(f"  BIASED:  {raw_dataset[i]['biased']}")
        print(f"  NEUTRAL: {raw_dataset[i]['neutral']}")

    # Transform to classification
    print("\n=== Classification Data (first 10 samples) ===")
    classification_dataset = transform_to_classification(raw_dataset)

    label_counts = {0: 0, 1: 0}
    for i in range(len(classification_dataset)):
        label_counts[classification_dataset[i]['label']] += 1

    print(f"Label distribution: {label_counts}")
    print(f"  0 (neutral): {label_counts[0]}")
    print(f"  1 (biased):  {label_counts[1]}")

    for i in range(10):
        sample = classification_dataset[i]
        label_name = "BIASED" if sample['label'] == 1 else "NEUTRAL"
        print(f"\nSample {i} [{label_name}]: {sample['text'][:100]}...")

    # Check tokenized data
    print("\n=== Tokenized Data (first 3 samples) ===")
    tokenizer = load_tokenizer()
    tokenized = get_classification_dataset(DatasetSplit.TRAIN, tokenizer)

    for i in range(3):
        sample = tokenized[i]
        print(f"\nSample {i}:")
        print(f"  labels: {sample['labels']}")
        print(f"  input_ids length: {len(sample['input_ids'])}")
        decoded = tokenizer.decode(sample['input_ids'][:50])
        print(f"  decoded (first 50 tokens): {decoded}")


if __name__ == "__main__":
    debug_dataset()
