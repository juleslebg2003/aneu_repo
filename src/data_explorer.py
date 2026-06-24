from pathlib import Path
import nibabel as nib
import pandas as pd
import json
import numpy as np
from collections import Counter

with open('/Users/julesperbet/aneu_repo/data/location_mapping.json', 'r') as f:
    data = json.load(f)

label_map = data['labels']
id_to_name = {v : k for k, v in label_map.items()}

class DataExplorer:

    def __init__(self, images_dir, labels_dir, label_map):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.id_to_name = label_map

        self.rows = []
        self.label_counter = Counter()

    def scan(self) : 
        for img_path in self.labels_dir.glob('*.nii.gz'):
            img = nib.load(img_path)
            shape = img.shape
            spacing = tuple(float(x) for x in img.header.get_zooms())
            
            mask = nib.load(img_path)
            mask_data = mask.get_fdata()
            values = np.unique(mask_data)
            values = values[values != 0]  # Exclude the background value (0)
            labels = [self.id_to_name[int(v)] for v in values]

            self.label_counter.update(labels)

            self.rows.append({
                'filename': img_path.name,
                'shape': shape,
                'spacing': spacing,
                'n_labels': len(labels),
                'labels': labels
             }) 
    
    def dataframe(self):
        df = pd.DataFrame(self.rows)
        df["filename"] = df["filename"].str.replace("_0000.nii.gz", ".nii.gz")
        return df
    
    def summary(self):

        df = self.dataframe()

        print("\n===== DATASET SUMMARY =====")

        print("\nNumber of scans:", len(df))
        print(df["shape"].value_counts().head())

        print("\nSpacing stats:")
        print(df["spacing"].value_counts().head())
        
        print("\nAverage labels per scan:", df["n_labels"].mean().round(2))

        print("\nLabel distribution:")
        for label, count in self.label_counter.items():
            print(f"{label}: {count}")
        
        print("\nMost common vessel labels:")
        for k, v in self.label_counter.most_common(5):
            print(f"{self.id_to_name[k]} : {v}")

explorer = DataExplorer(
    images_dir="/Users/julesperbet/aneu_repo/data/images",
    labels_dir="/Users/julesperbet/aneu_repo/data/location_masks",
    label_map=id_to_name
)

explorer.scan()

df = explorer.dataframe()

explorer.summary()

print(df.head())