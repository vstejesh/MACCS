import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataset import MultiOmicsDataset
from model import GeneralizedCrossAttn 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import copy

from sklearn.metrics import (accuracy_score, f1_score, classification_report, 
                             confusion_matrix, roc_auc_score, roc_curve, auc)
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = "imputed_combined_omics_383.npz"  
BATCH_SIZE = 16                     
LEARNING_RATE = 1e-4            
NUM_EPOCHS = 10
NUM_CLASSES = 2                     
NUM_FOLDS = 5                      # Number of cross-validation folds
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CENTRAL_OMIC = "mRNA" 

def main():
    # Adding a global seed for PyTorch DataLoaders (shuffle=True) 
    # to ensure batches are drawn in the exact same order every run.
    if CENTRAL_OMIC == "mRNA":
        seed_val = 41
    elif CENTRAL_OMIC == "miRNA":
        seed_val = 44
    elif CENTRAL_OMIC == "CNV":
        seed_val = 34
    else:
        seed_val = 47        
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)    

    print(f"Using device: {DEVICE}")

    # 1. LOAD DATA
    # ---------------------------
    print("Loading dataset...")
    full_dataset = MultiOmicsDataset(DATA_PATH, omics=["mRNA", "Methylation", "CNV", "miRNA"])
    all_labels = full_dataset.labels.cpu().numpy()

    # Data structures to store Out-Of-Fold (OOF) predictions
    oof_preds = []
    oof_probs = []
    oof_labels = []
    oof_latents = []
    
    best_macro_f1 = 0.0
    best_model_state = None
    best_val_loader = None
    best_fold_idx = 0
    fold_accs = []
    fold_mac_f1s = []
    fold_wei_f1s = []
    fold_roc_aucs = []
    fold_sensitivities = []
    fold_specificities = []

    # Stratified K-Fold Setup
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    print("\nSaving initial model weights for exact fold replication...")
    feature_sizes = {
        'mRNA': 383,
        'Methylation': 383,
        'CNV': 383,
        'miRNA': 383
    }
    # Initialize a base model purely to capture its initial random state
    base_model = GeneralizedCrossAttn(num_classes=NUM_CLASSES, central_omic=CENTRAL_OMIC, feature_sizes=feature_sizes).to(DEVICE)
    torch.save(base_model.state_dict(), 'initial_weights.pth')
    del base_model # Clean up memory, we only needed the saved file

    print(f"\nStarting {NUM_FOLDS}-Fold Cross Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels)):
        print(f"\n{'='*40}")
        print(f" FOLD {fold + 1}/{NUM_FOLDS}")
        print(f"{'='*40}")

        # ==========================================
        # DATASET SIZE & DISTRIBUTION INFO
        # ==========================================
        val_size = len(val_idx)
        train_size = len(train_idx)
        
        # Get unique classes and their counts in the validation set
        val_classes, val_counts = np.unique(all_labels[val_idx], return_counts=True)
        
        print(f"Train Size: {train_size} | Val (Test) Size: {val_size}")
        print("Validation Class Distribution:")
        for cls, count in zip(val_classes, val_counts):
            pct = (count / val_size) * 100
            print(f"  -> Class {int(cls)}: {count} samples ({pct:.1f}%)")
        print('-' * 40)

        # Create Fold Subsets
        train_dataset = MultiOmicsDataset.from_subset(full_dataset, train_idx)
        val_dataset = MultiOmicsDataset.from_subset(full_dataset, val_idx)

        # APPLY SMOTE STRICTLY TO TRAIN SUBSET
        train_dataset.apply_smote()

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize fresh model for this specific fold
        model = GeneralizedCrossAttn(num_classes=NUM_CLASSES, central_omic=CENTRAL_OMIC, feature_sizes=feature_sizes).to(DEVICE)
       
        model.load_state_dict(torch.load('initial_weights.pth'))
        print("-> Loaded exact initial starting weights for this fold.")

        criterion = nn.CrossEntropyLoss().to(DEVICE)
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)

        # --- FOLD TRAINING LOOP ---
        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                labels = labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                if isinstance(outputs, tuple): outputs = outputs[0]
                
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()

            # --- FOLD VALIDATION LOOP ---
            model.eval()
            val_loss = 0.0
            fold_val_preds = []
            fold_val_labels = []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                    labels = labels.to(DEVICE)
                    
                    outputs = model(inputs)
                    if isinstance(outputs, tuple): outputs = outputs[0]
                        
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    fold_val_preds.extend(predicted.cpu().numpy())
                    fold_val_labels.extend(labels.cpu().numpy())
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            fold_acc = 100 * accuracy_score(fold_val_labels, fold_val_preds)
            
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {fold_acc:.2f}%")

        # --- END OF FOLD EXTRACTION ---
        model.eval()
        fold_probs = []
        fold_latents = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                outputs, fused = model(inputs, return_latent=True)
                
                # Softmax to get probabilities for Class 1 (for ROC-AUC)
                probs = torch.softmax(outputs, dim=1)[:, 1] 
                
                fold_probs.extend(probs.cpu().numpy())
                fold_latents.append(fused.cpu().numpy())

        # Track Out-of-Fold data
        oof_preds.extend(fold_val_preds)
        oof_probs.extend(fold_probs)
        oof_labels.extend(fold_val_labels)
        oof_latents.append(np.concatenate(fold_latents, axis=0))

        # ==========================================
        # CALCULATE ALL METRICS FOR THE CURRENT FOLD
        # ==========================================
        fold_val_labels_np = np.array(fold_val_labels)
        fold_val_preds_np = np.array(fold_val_preds)
        fold_probs_np = np.array(fold_probs)

        fold_acc = accuracy_score(fold_val_labels_np, fold_val_preds_np)
        fold_mac_f1 = f1_score(fold_val_labels_np, fold_val_preds_np, average='macro')
        fold_wei_f1 = f1_score(fold_val_labels_np, fold_val_preds_np, average='weighted')
        
        if len(np.unique(fold_val_labels_np)) > 1:
            fold_roc_auc = roc_auc_score(fold_val_labels_np, fold_probs_np)
        else:
            fold_roc_auc = float('nan')

        fold_cm = confusion_matrix(fold_val_labels_np, fold_val_preds_np, labels=[0, 1])
        if fold_cm.shape == (2, 2):
            tn, fp, fn, tp = fold_cm.ravel()
            fold_sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fold_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            fold_sens, fold_spec = 0.0, 0.0

        # Print Fold Metrics
        print(f"\n--- Fold {fold+1} Evaluation ---")
        print(f"Accuracy:    {100 * fold_acc:.2f}%")
        print(f"Macro F1:    {fold_mac_f1:.4f}")
        print(f"Weighted F1: {fold_wei_f1:.4f}")
        print(f"ROC-AUC:     {fold_roc_auc:.4f}")
        print(f"Sensitivity: {fold_sens:.4f} (Recall Class 1)")
        print(f"Specificity: {fold_spec:.4f} (True Negative Rate)")

        fold_accs.append(fold_acc)
        fold_mac_f1s.append(fold_mac_f1)
        fold_wei_f1s.append(fold_wei_f1)
        fold_roc_aucs.append(fold_roc_auc)
        fold_sensitivities.append(fold_sens)
        fold_specificities.append(fold_spec)

        if fold_mac_f1 > best_macro_f1:
            best_macro_f1 = fold_mac_f1
            best_model_state = copy.deepcopy(model.state_dict())
            best_val_loader = val_loader
            best_fold_idx = fold + 1

    print(f"\n{'='*40}")
    print(f" CROSS VALIDATION COMPLETE ")
    print(f" Average Macro F1: {np.mean(fold_mac_f1s):.4f} ± {np.std(fold_mac_f1s):.4f}")
    print(f" Average Accuracy: {np.mean(fold_accs):.2f}% ± {np.std(fold_accs):.2f}%")
    print(f" Average Weighted F1: {np.mean(fold_wei_f1s):.4f} ± {np.std(fold_wei_f1s):.4f}")
    print(f" Average ROC-AUC: {np.nanmean(fold_roc_aucs):.4f} ± {np.nanstd(fold_roc_aucs):.4f}")
    print(f" Average Sensitivity: {np.mean(fold_sensitivities):.4f} ± {np.std(fold_sensitivities):.4f}")
    print(f" Average Specificity: {np.mean(fold_specificities):.4f} ± {np.std(fold_specificities):.4f}")
    print(f"{'='*40}")

    # ==========================================
    # FINAL METRICS EVALUATION (Aggregated OOF)
    # ==========================================
    oof_labels = np.array(oof_labels)
    oof_preds = np.array(oof_preds)
    oof_probs = np.array(oof_probs)
    oof_latents = np.concatenate(oof_latents, axis=0)

    cm = confusion_matrix(oof_labels, oof_preds)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    roc_auc = roc_auc_score(oof_labels, oof_probs)

    print("\n--- Aggregated Classification Metrics (Out-of-Fold) ---")
    print(f"Accuracy:    {100 * accuracy_score(oof_labels, oof_preds):.2f}%")
    print(f"Macro F1:    {f1_score(oof_labels, oof_preds, average='macro'):.4f}")
    print(f"Weighted F1: {f1_score(oof_labels, oof_preds, average='weighted'):.4f}")
    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f} (Recall for Class 1)")
    print(f"Specificity: {specificity:.4f} (True Negative Rate)")
    print("\nClassification Report:")
    print(classification_report(oof_labels, oof_preds))

    # Plots
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(oof_labels), yticklabels=np.unique(oof_labels))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Aggregated Cross-Validation Confusion Matrix')
    plt.savefig('confusion_matrix_cv.png')
    plt.close()

    print("Generating ROC Curve...")
    fpr, tpr, thresholds = roc_curve(oof_labels, oof_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (Aggregated OOF)')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_cv.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================================
    # PCA LATENT SPACE VISUALIZATION (95% Target)
    # ==========================================
    print("\n--- Running PCA Dimensionality Reduction ---")
    
    pca_95 = PCA(n_components=0.95, random_state=42)
    pca_95.fit(oof_latents)
    
    n_components_95 = pca_95.n_components_
    actual_var_captured = np.sum(pca_95.explained_variance_ratio_)
    
    print(f"Number of PCs required to capture ≥ 95% variance: {n_components_95}")
    print(f"Actual variance captured by these {n_components_95} PCs: {actual_var_captured:.2%}")

    if len(pca_95.explained_variance_ratio_) >= 4:
        print("\nVariance Explained by first 4 Principal Components:")
        for i in range(4):
            print(f"  PC{i+1}: {pca_95.explained_variance_ratio_[i]:.2%}")

    pca_2d = PCA(n_components=2, random_state=42)
    latent_2d = pca_2d.fit_transform(oof_latents)
    var_2d = pca_2d.explained_variance_ratio_

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=latent_2d[:, 0], y=latent_2d[:, 1], hue=oof_labels, palette="Set1", s=100, alpha=0.8)
    plt.title(f'Latent Space (PCA)\nPC1: {var_2d[0]:.1%} | PC2: {var_2d[1]:.1%} variance\n({n_components_95} PCs needed for 95% total variance)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='True Class')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('latent_space_pca.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================================
    # MODEL INTERPRETABILITY (On the best fold)
    # ==========================================
    print(f"\nLoading best model (from Fold {best_fold_idx}) for interpretability analysis...")
    best_model = GeneralizedCrossAttn(num_classes=NUM_CLASSES, central_omic=CENTRAL_OMIC, feature_sizes=feature_sizes).to(DEVICE)
    best_model.load_state_dict(best_model_state)
    
    best_model.eval()
    best_fold_preds = []
    best_fold_labels = []
    with torch.no_grad():
        for inputs, labels in best_val_loader:
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = best_model(inputs)
            if isinstance(outputs, tuple): outputs = outputs[0]
            _, predicted = torch.max(outputs.data, 1)
            best_fold_preds.extend(predicted.cpu().numpy())
            best_fold_labels.extend(labels.cpu().numpy())

    omics_list = ["mRNA", "Methylation", "CNV", "miRNA"]
    torch.save(best_model.state_dict(), "best_kfold_model.pth")
    print("\nSaved the highest-performing fold model to 'best_kfold_model.pth'. Process complete!")

if __name__ == "__main__":
    main()