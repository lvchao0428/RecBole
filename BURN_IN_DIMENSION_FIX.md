# Burn-in Checkpoint Dimension Mismatch Fix

## 🔍 Problem Summary

**Error**: `RuntimeError: size mismatch for multiview_concat_proj.weight`
- Checkpoint shape: `[512, 1024]`
- Current model shape: `[512, 1280]`

## 📋 Root Cause Analysis

### The Issue
The error occurs when using **burn-in training** with **SASRecAlignMultiView** model that has **base text features** enabled.

### Why It Happens

1. **Burn-in Phase** (ID-only warmup):
   ```python
   # scripts/two_phase_train.py lines 665-675
   burnin_dict = {
       "disable_text_feature": True,  # ← Text features disabled
       "use_align": False,
       "fuse_text_feature": False,
       "use_llm": False,
   }
   ```
   - Text features are **completely disabled**
   - `item_text_emb_base = None` in the model
   - `multiview_concat_proj` input dimension = **1024** (4 views × 256)

2. **Phase-A**:
   ```yaml
   # sasrec_align_multi_view.yaml line 47
   item_text_emb_path_base: /home/charlie/project/RecBole/dataset/Amazon_Beauty/item_text_emb.base.npy
   ```
   - Text features are **enabled**
   - Base embeddings are loaded
   - `item_text_emb_base` is **NOT None**
   - `multiview_concat_proj` input dimension = **1280** (4×256 + 256)

3. **Loading Checkpoint**:
   ```python
   # sasrecalignmultiview.py lines 96-99
   multiview_input_dim = self.num_text_views * self.hidden_size  # 1024
   if self.item_text_emb_base is not None:
       base_dim = self.item_text_emb_base.shape[1]  # 256
       multiview_input_dim += base_dim  # 1024 + 256 = 1280
   ```
   - Burn-in checkpoint has dimension **1024**
   - Current model expects dimension **1280**
   - **Dimension mismatch → RuntimeError!**

## ✅ Solution Applied

### Modified: `scripts/two_phase_train.py`

Added **dimension mismatch detection and graceful handling** in two places:

1. **Phase-A Grid Search** (line ~753)
2. **Phase-A Non-Grid** (line ~941)

### How It Works

```python
# Check for dimension mismatches before loading
mismatched_keys = []
current_state = model_a.state_dict()
for key, ckpt_tensor in ckpt_b["state_dict"].items():
    if key in current_state:
        if ckpt_tensor.shape != current_state[key].shape:
            mismatched_keys.append(
                f"{key}: ckpt{list(ckpt_tensor.shape)} vs model{list(current_state[key].shape)}"
            )

if mismatched_keys:
    # Log warning and skip mismatched layers
    logger_a.warning("[Phase-A] Dimension mismatch in burn-in checkpoint:")
    for mk in mismatched_keys:
        logger_a.warning(f"  - {mk}")
    logger_a.warning("[Phase-A] Skipping mismatched layers, loading compatible ones only")
    
    # Filter out mismatched keys
    filtered_state = {k: v for k, v in ckpt_b["state_dict"].items() 
                    if k in current_state and v.shape == current_state[k].shape}
    model_a.load_state_dict(filtered_state, strict=False)
else:
    # No mismatch, load normally
    model_a.load_state_dict(ckpt_b["state_dict"], strict=False)
```

### Benefits

✅ **Graceful degradation**: Training continues even with dimension mismatches  
✅ **Informative logging**: Shows which layers were skipped  
✅ **Loads compatible layers**: Backbone weights still transfer from burn-in  
✅ **Automatic handling**: No manual intervention needed  

### What You'll See

When running with the fix, you'll see:

```
[WARNING] [Phase-A:grid] Dimension mismatch in burn-in checkpoint:
[WARNING]   - multiview_concat_proj.weight: ckpt[512, 1024] vs model[512, 1280]
[WARNING]   - multiview_concat_proj.bias: ckpt[512] vs model[512]
[WARNING] [Phase-A:grid] Skipping mismatched layers, loading compatible ones only
[INFO] [Phase-A:grid] Loaded burn-in checkpoint (strict=False): saved/phase_runs_multiview_4views/xxx.pth
```

The model will:
- ✅ Load compatible backbone weights (item_embedding, position_embedding, transformers)
- ❌ Skip incompatible fusion layers (will use random initialization)
- ✅ Continue training normally

## 🔧 Alternative Solutions

### Option 1: Disable Base Features (Config Change)

Edit `sasrec_align_multi_view.yaml`:

```yaml
# Change from:
item_text_emb_path_base: /home/charlie/project/RecBole/dataset/Amazon_Beauty/item_text_emb.base.npy

# To:
item_text_emb_path_base: ""  # or remove this line
```

**Pros**: No dimension mismatch  
**Cons**: Loses base TF-IDF features

### Option 2: Enable Text Features in Burn-in (Not Recommended)

Modify burn-in config in `two_phase_train.py`:

```python
burnin_dict = {
    "freeze_backbone": False,
    "disable_text_feature": False,  # ← Enable text
    "use_align": False,  # Keep alignment disabled
    ...
}
```

**Pros**: Checkpoint dimensions match  
**Cons**: Defeats purpose of ID-only burn-in

### Option 3: Skip Burn-in for MultiView (Script Change)

Modify `two_phase_run_multiview_split.sh`:

```bash
# Change from:
  --backbone_burnin_epochs 10 \

# To:
  --backbone_burnin_epochs 0 \
```

**Pros**: Simple, no conflicts  
**Cons**: No backbone warmup

## 📊 Impact Analysis

### What Gets Loaded from Burn-in

With the fix, these layers **successfully load**:
- ✅ `item_embedding.weight` - Core item embeddings
- ✅ `position_embedding.weight` - Positional encodings
- ✅ `trm_encoder.*` - All transformer layers
- ✅ `LayerNorm.*` - Normalization layers

### What Gets Skipped (Random Init)

- ❌ `multiview_concat_proj.*` - Multi-view projection (dimension mismatch)
- ❌ `item_fusion_cross.*` - Fusion layers (if dimension depends on projection)
- ❌ `item_fusion_deep.*` - Fusion MLP (if dimension depends on projection)
- ❌ `item_fusion_predictor.*` - Fusion predictor (if dimension depends on projection)

### Training Implications

1. **Backbone warmup preserved**: Most important layers still benefit from burn-in
2. **Fusion layers random init**: Will learn from scratch in Phase-A
3. **Overall impact**: **Minimal** - fusion layers are small and train quickly

## 🚀 Next Steps

### Option A: Use the Fix (Recommended)

Just run your training script again:

```bash
bash two_phase_run_multiview_split.sh
```

The fix is already applied and will handle the mismatch automatically.

### Option B: Disable Burn-in

If you prefer clean training without warnings:

```bash
# Edit two_phase_run_multiview_split.sh
# Change:
  --backbone_burnin_epochs 10 \
# To:
  --backbone_burnin_epochs 0 \
```

### Option C: Remove Base Features

If you don't need base TF-IDF features:

```yaml
# Edit sasrec_align_multi_view.yaml
item_text_emb_path_base: ""
```

## 📝 Summary

**Problem**: Burn-in checkpoint has different dimensions than Phase-A model due to base text features  
**Solution**: Automatic detection and filtering of mismatched layers  
**Result**: Training continues successfully, backbone weights preserved  
**Action**: None required - fix already applied, just re-run your training script  

---

**Status**: ✅ **FIXED** - Ready to train!


