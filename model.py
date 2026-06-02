import torch
import torch.nn as nn

class MicroEncoder(nn.Module):
    def __init__(self, in_features=128, out_features=16):
        super().__init__()
        # ONE flat layer. No hidden dimensions. No ReLUs.
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(0.4) # Emulating colsample_bytree

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        return self.dropout(x)

class MicroCrossAttention(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        # A single, tiny attention head
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)

    def forward(self, target_omic, source_omic):
        # target_omic (mRNA) is Query
        # source_omic (Methylation) is Key and Value
        
        # Reshape for attention: (Batch, Seq_Len=1, Features=16)
        k=v=target_omic.unsqueeze(1) 
        q = source_omic.unsqueeze(1)
        
        attn_output, _ = self.attention(q, k, v)
        attn_output = attn_output.squeeze(1)  # Back to (Batch, 16)
        return attn_output  # Residual connection

class GeneralizedCrossAttn(nn.Module):
    def __init__(self, central_omic='mRNA', modalities=None, feature_sizes=None, num_classes=2, hidden_dim=16):
        super().__init__()
        
        # Default modalities if none are provided
        if modalities is None:
            self.modalities = ['mRNA', 'Methylation', 'CNV', 'miRNA']
        else:
            self.modalities = modalities
            
        self.central_omic = central_omic
        
        # Safety check for central omic
        if self.central_omic not in self.modalities:
            raise ValueError(f"Central omic '{self.central_omic}' must be one of the defined modalities: {self.modalities}")
        
        # Handle feature sizes dictionary
        if feature_sizes is None:
            # Fallback: assume 383 for all modalities if nothing is passed
            self.feature_sizes = {mod: 383 for mod in self.modalities}
        else:
            self.feature_sizes = feature_sizes
            # Safety check to ensure every modality has a specified input size
            for mod in self.modalities:
                if mod not in self.feature_sizes:
                    raise ValueError(f"Missing input size for '{mod}' in the feature_sizes dictionary.")

        # 1. Dynamic Encoders (Now with modality-specific input dimensions)
        self.encoders = nn.ModuleDict({
            mod: MicroEncoder(self.feature_sizes[mod], hidden_dim) for mod in self.modalities
        })
        
        # 2. Dynamic Attention Blocks
        # Create a cross-attention block for every modality projecting INTO the central omic
        self.attentions = nn.ModuleDict({
            mod: MicroCrossAttention(hidden_dim) for mod in self.modalities if mod != self.central_omic
        })
        
        # 3. Classifier
        # Total concatenated features = (number of modalities) * hidden_dim
        concat_dim = len(self.modalities) * hidden_dim 
        
        self.classifier = nn.Sequential(
            # First Hidden Block
            # nn.Linear(concat_dim, 128),
            # nn.BatchNorm1d(128),      
            # nn.ReLU(),                
            # nn.Dropout(p=0.4),        
            
            # Second Hidden Block
            nn.Linear(concat_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.3),        
            
            # Final Output Layer
            nn.Linear(64, num_classes)
        )

    def forward(self, x, return_latent=False):
        # 1. Encode all available modalities dynamically
        encoded_feats = {}
        for mod in self.modalities:
            # Passes x[mod] through the encoder specifically sized for it
            encoded_feats[mod] = self.encoders[mod](x[mod])
            
        # 2. Attend (Star Topology into the central_omic)
        central_feat = encoded_feats[self.central_omic]
        
        # We start our fused list with the central feature
        fused_list = [central_feat]
        
        for mod in self.modalities:
            if mod != self.central_omic:
                # Query is the central_omic, Key/Value is the other omic
                attended_feat = self.attentions[mod](central_feat, encoded_feats[mod])
                fused_list.append(attended_feat)
        
        # 3. Fuse and Classify
        fused = torch.cat(fused_list, dim=1)
        out = self.classifier(fused)

        if return_latent:
            return out, fused
        
        return out