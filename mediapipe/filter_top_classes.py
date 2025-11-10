import pickle
import argparse
from collections import Counter

def filter_top_classes(features_file, output_file, top_k=25):
    with open(features_file, 'rb') as f:
        data = pickle.load(f)
    # Count class frequencies
    label_counter = Counter(info['label'] for info in data.values())
    # Get top-K frequent class labels
    top_labels = set(label for label, _ in label_counter.most_common(top_k))
    print(f"Top-{top_k} labels: {sorted(top_labels)}")
    # Build label remap: original_label → new_label (0 to top_k-1)
    old_to_new = {orig: i for i, orig in enumerate(sorted(top_labels))}
    filtered = {}
    kept = 0
    for vid, info in data.items():
        if info['label'] in top_labels:
            new_label = old_to_new[info['label']]
            filtered[vid] = dict(info)  # shallow copy
            filtered[vid]['label'] = new_label
            kept += 1
    print(f"Kept {kept} samples from {len(filtered)} videos, {top_k} classes.")
    with open(output_file, 'wb') as f:
        pickle.dump(filtered, f)
    print(f"Filtered data saved as: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--top_k', type=int, default=25)
    args = parser.parse_args()
    filter_top_classes(args.features_file, args.output_file, args.top_k)
