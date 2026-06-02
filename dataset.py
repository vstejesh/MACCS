import torch
import numpy as np
from torch.utils.data import Dataset
from imblearn.over_sampling import SMOTE

class MultiOmicsDataset(Dataset):
    def __init__(self, data_path, omics):
        self.data = np.load(data_path)
        self.omics = omics
        self.labels = torch.from_numpy(self.data["y"]).long()
        
        # --- NEW: Calculate Mean and Std for scaling ---
        self.stats = {}
        print("Calculating normalization stats...")
        for omic in omics:
            # We convert to float32 immediately to save memory/time
            raw = self.data[omic].astype(np.float32)
            
            # Calculate stats across all patients (axis 0)
            mean = np.mean(raw, axis=0)
            std = np.std(raw, axis=0)
            
            # Avoid division by zero: if std is 0, set it to 1
            std[std == 0] = 1.0
            
            self.stats[omic] = {
                "mean": torch.from_numpy(mean),
                "std": torch.from_numpy(std)
            }
        print("Normalization stats ready.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = {}
        for omic in self.omics:
            # Load raw data
            tensor = torch.from_numpy(self.data[omic][index, :]).float()
            
            # --- APPLY NORMALIZATION: (Value - Mean) / Std ---
            tensor = (tensor - self.stats[omic]["mean"]) / self.stats[omic]["std"]
            
            # Safety check: Replace any accidental NaNs with 0
            tensor = torch.nan_to_num(tensor, nan=0.0)
            
            sample[omic] = tensor
            
        return sample, self.labels[index]
    
    def apply_smote(self):
        print("Applying SMOTE to balance classes...")
        # 1. Concatenate all modalities into one large matrix for resampling
        # Shape: (284, 1600) -> (4 modalities * 400 features)
        combined_features = np.concatenate([self.data[o] for o in self.omics], axis=1)
        
        smote = SMOTE(random_state=42)
        features_resampled, labels_resampled = smote.fit_resample(combined_features, self.labels)
        
        # 2. Split the combined matrix back into individual modalities
        new_data = {}
        # for i, omic in enumerate(self.omics):
        #     start_col = i * 400
        #     end_col = (i + 1) * 400
        #     new_data[omic] = features_resampled[:, start_col:end_col]
        num_features = self.data[self.omics[0]].shape[1] # Automatically detects 383
        for i, omic in enumerate(self.omics):
            start_col = i * num_features
            end_col = (i + 1) * num_features
            new_data[omic] = features_resampled[:, start_col:end_col]
            
        self.data = new_data
        self.labels = torch.from_numpy(labels_resampled).long()
        print(f"New Dataset Size: {len(self.labels)} (Balanced)")

    @classmethod    
    def from_subset(cls, parent_dataset, indices):
        """
        Creates a new MultiOmicsDataset from specific indices (e.g., from a split).
        This allows us to safely use apply_smote() on just the training subset.
        """
        # Create an empty instance bypassing __init__
        new_dataset = cls.__new__(cls)
        new_dataset.omics = parent_dataset.omics
        
        # 1. Extract only the subset data arrays
        new_dataset.data = {}
        for omic in new_dataset.omics:
            new_dataset.data[omic] = parent_dataset.data[omic][indices]
        
        # 2. Extract labels
        new_dataset.labels = parent_dataset.labels[indices]
        
        # 3. Inherit normalization stats so Train and Val use the exact same scaling!
        if hasattr(parent_dataset, 'stats'):
            new_dataset.stats = parent_dataset.stats
            
        return new_dataset