
import torch
import sys

# Copy of the function to test logic in isolation
def remap_state_dict(state_dict):
    """
    Remap state_dict keys to be compatible with the current environment.
    Specifically handles Flash Attention (checkpoint) -> Standard Attention (local) conversion.
    """
    new_state_dict = {}
    
    # Group Wq and Wkv keys
    attention_groups = {}
    
    for key, value in state_dict.items():
        # Handle Flash Attention Wq/Wkv -> in_proj
        if ".attention.Wq." in key:
            parts = key.split(".attention.Wq.")
            prefix = parts[0] + ".attention."
            suffix = parts[1] # weight or bias
            
            if prefix not in attention_groups:
                attention_groups[prefix] = {}
            attention_groups[prefix][f"Wq_{suffix}"] = value
            continue
            
        if ".attention.Wkv." in key:
            parts = key.split(".attention.Wkv.")
            prefix = parts[0] + ".attention."
            suffix = parts[1] # weight or bias
            
            if prefix not in attention_groups:
                attention_groups[prefix] = {}
            attention_groups[prefix][f"Wkv_{suffix}"] = value
            continue
            
        # Handle out_proj path mismatch
        # Checkpoint: ...attention.out_proj.weight
        # Target: ...attention.attn.out_proj.weight
        if ".attention.out_proj." in key:
            new_key = key.replace(".attention.out_proj.", ".attention.attn.out_proj.")
            new_state_dict[new_key] = value
            continue
            
        # Keep other keys as is
        new_state_dict[key] = value
        
    # Process grouped attention weights
    for prefix, groups in attention_groups.items():
        # We expect Wq_weight, Wkv_weight, Wq_bias, Wkv_bias
        
        # Handle Weights
        if "Wq_weight" in groups and "Wkv_weight" in groups:
            wq = groups["Wq_weight"]
            wkv = groups["Wkv_weight"]
            # Wq: (embed_dim, embed_dim)
            # Wkv: (2*embed_dim, embed_dim)
            # in_proj_weight: (3*embed_dim, embed_dim) -> [q, k, v]
            
            in_proj_weight = torch.cat([wq, wkv], dim=0)
            new_key = prefix + "attn.in_proj_weight"
            new_state_dict[new_key] = in_proj_weight
            
        # Handle Biases
        if "Wq_bias" in groups and "Wkv_bias" in groups:
            bq = groups["Wq_bias"]
            bkv = groups["Wkv_bias"]
            
            in_proj_bias = torch.cat([bq, bkv], dim=0)
            new_key = prefix + "attn.in_proj_bias"
            new_state_dict[new_key] = in_proj_bias
            
    return new_state_dict

def test_remapping():
    print("Testing remap_state_dict...")
    
    embed_dim = 32
    
    # Simulate Flash Attention weights
    # Wq: (embed_dim, embed_dim)
    # Wkv: (2*embed_dim, embed_dim)
    # out_proj: (embed_dim, embed_dim)
    
    wq_weight = torch.randn(embed_dim, embed_dim)
    wq_bias = torch.randn(embed_dim)
    wkv_weight = torch.randn(2 * embed_dim, embed_dim)
    wkv_bias = torch.randn(2 * embed_dim)
    out_proj_weight = torch.randn(embed_dim, embed_dim)
    out_proj_bias = torch.randn(embed_dim)
    
    state_dict = {
        "model.bone_encoder.attn.0.attention.Wq.weight": wq_weight,
        "model.bone_encoder.attn.0.attention.Wq.bias": wq_bias,
        "model.bone_encoder.attn.0.attention.Wkv.weight": wkv_weight,
        "model.bone_encoder.attn.0.attention.Wkv.bias": wkv_bias,
        "model.bone_encoder.attn.0.attention.out_proj.weight": out_proj_weight,
        "model.bone_encoder.attn.0.attention.out_proj.bias": out_proj_bias,
        "other.param": torch.tensor([1.0])
    }
    
    new_state_dict = remap_state_dict(state_dict)
    
    # Verify keys
    expected_keys = [
        "model.bone_encoder.attn.0.attention.attn.in_proj_weight",
        "model.bone_encoder.attn.0.attention.attn.in_proj_bias",
        "model.bone_encoder.attn.0.attention.attn.out_proj.weight",
        "model.bone_encoder.attn.0.attention.attn.out_proj.bias",
        "other.param"
    ]
    
    for key in expected_keys:
        if key not in new_state_dict:
            print(f"FAILED: Missing key {key}")
            # Print available keys to debug
            print("Available keys:", list(new_state_dict.keys()))
            return
            
    # Verify shapes
    in_proj_weight = new_state_dict["model.bone_encoder.attn.0.attention.attn.in_proj_weight"]
    if in_proj_weight.shape != (3 * embed_dim, embed_dim):
        print(f"FAILED: in_proj_weight shape mismatch. Got {in_proj_weight.shape}, expected {(3 * embed_dim, embed_dim)}")
        return
        
    # Verify values
    # Check if Wq part matches
    if not torch.allclose(in_proj_weight[:embed_dim], wq_weight):
        print("FAILED: Wq part of in_proj_weight mismatch")
        return
        
    # Check if Wkv part matches
    if not torch.allclose(in_proj_weight[embed_dim:], wkv_weight):
        print("FAILED: Wkv part of in_proj_weight mismatch")
        return

    print("SUCCESS: State dict remapping verified!")

if __name__ == "__main__":
    test_remapping()
