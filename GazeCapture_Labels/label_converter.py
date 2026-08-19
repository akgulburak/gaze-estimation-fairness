import pandas as pd
from pathlib import Path

# Paths
main_csv_path = "test.label"          # first csv file
labels_csv_path = "gazecapture_dataset_gender_v5.csv"      # dataframe csv file
output_csv_path = "test_with_gender.label"

# Read files
# Your first file looks space-separated, not comma-separated
main_df = pd.read_csv(main_csv_path, sep=r"\s+")

labels_df = pd.read_csv(labels_csv_path)

# Extract id from Face path
# Example:
# Image/test/Face/00010_00000.jpg -> 00010_00000.jpg -> 00010 -> 10
def extract_id_from_face_path(face_path):
    filename = Path(face_path).name          # 00010_00000.jpg
    id_part = filename.split("_")[0]         # 00010
    return int(id_part)                      # 10

main_df["id"] = main_df["Face"].apply(extract_id_from_face_path)

# Create id -> group mapping
id_to_group = labels_df.set_index("id")["group"]

# Add demographic label
main_df["DemographicLabel"] = main_df["id"].map(id_to_group)

# Remove helper id column if you do not want it in the final file
main_df = main_df.drop(columns=["id"])

# Save output
main_df.to_csv(output_csv_path, sep=" ", index=False)

print(f"Saved to {output_csv_path}")