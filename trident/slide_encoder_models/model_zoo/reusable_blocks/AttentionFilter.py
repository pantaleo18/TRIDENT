import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFilter(nn.Module):
    """
    Versione semplificata di ABMIL:
    - 1 testa
    - 1 branch
    - Restituisce features con peso di attenzione applicato
    """
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float = 0.0, gated: bool = False):
        super().__init__()
        self.gated = gated

        # Linear per calcolare le features di attenzione
        self.attention_vector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

        # Se vogliamo gating, calcola fattori sigmoid
        if self.gated:
            self.gating_vector = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.Sigmoid(),
                nn.Dropout(dropout)
            )

        # Layer finale per calcolare il peso della patch (1 branch)
        self.branch_layer = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            features: (B, N_patches, feature_dim)
            attn_mask: (B, N_patches) opzionale, 1=patch valida, 0=da ignorare

        Returns:
            weighted_features: (B, N_patches, hidden_dim), features moltiplicate per peso di attenzione
        """
        # Attention vector
        attn_vec = self.attention_vector(features)  # (B, N, hidden_dim)

        # Gating opzionale
        if self.gated:
            gate = self.gating_vector(features)       # (B, N, hidden_dim)
            attn_vec = attn_vec * gate                # elemento per elemento

        # Peso scalare per patch
        patch_score = self.branch_layer(attn_vec)     # (B, N, 1)

        # Applica maschera se fornita
        if attn_mask is not None:
            patch_score = patch_score.masked_fill(~attn_mask.unsqueeze(-1), float('-inf'))

        # Softmax lungo le patch (attenzione tra patch)
        patch_weight = F.softmax(patch_score, dim=1)  # (B, N, 1)

        # Moltiplica features per peso di attenzione
        weighted_features =  features * patch_weight   # (B, N, feature_dim)

        return weighted_features
