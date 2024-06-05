import torch
import einops

# Sample tensors with the specified requires_grad properties
resid_post = torch.randn(10, 20, 30, requires_grad=False)
linear_probe = torch.randn(5, 30, 10, 10, 3, requires_grad=True)

# Using einops (assuming the einsum operation is correct and you want to keep using einops)
'''probe_out = einops.einsum(
    resid_post,
    linear_probe,
    "batch pos d_model, modes d_model rows cols options -> modes batch pos rows cols options"
)'''
probe_out = einops.einsum(
    resid_post,
    linear_probe,
    "batch pos d_model, modes d_model rows cols options -> modes batch pos rows cols options",
)

# Check requires_grad
print(probe_out.requires_grad)  # Should print True