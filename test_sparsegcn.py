import torch
from kaggle_notebooks.sparsegcn_carm import SparseGCNCARMModel
cfg = {
  'topk_k': 8,
  'lambda_feat': 0.3,
  'hop_alpha': 0.5,
  'edge_dropout': 0.0,
  'use_pairnorm': True,
  'use_residual': True,
  'use_channel_attention': True,
  'attention_heads': 4,
  'temporal_scales': [8,16,32]
}
C,T,K,H = 64, 256, 2, 40
m = SparseGCNCARMModel(C,T,K,H,cfg)
x = torch.randn(4,1,C,T)
logits = m(x)
print('OK logits:', logits.shape)
# simulate pruning
m.channel_importance = torch.rand(C)
pruned = m.prune_channels(0.1, min_channels=20)
print('Pruned', pruned)
logits = m(x)
print('OK logits after prune:', logits.shape)
print('Adj type:', type(m.get_final_adjacency()).__name__)
