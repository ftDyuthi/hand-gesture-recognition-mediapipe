import pandas as pd

df = pd.read_csv("../../../top25_videoid_label_map.csv")
labels = sorted(df["label"].unique())
print("Number of classes:", len(labels))
print(labels)

# Save final label ordering
with open("/workspace/top25_labels.txt", "w") as f:
    for lbl in labels:
        f.write(lbl + "\n")

print("✅ Saved /workspace/top25_labels.txt")
