class FeatureHookManager:
    def __init__(self):
        self.features = {}
        self.hooks = []

    def hook_fn(self, name):
        def hook(module, input, output):
            self.features[name] = output
        return hook

    def register_hook(self, module, name):
        hook = module.register_forward_hook(self.hook_fn(name))
        self.hooks.append(hook)

    def clear(self):
        self.features.clear()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

import torch
import torch.nn.functional as F

class AttentionHookManager:
    def __init__(self):
        self.attention_maps = {}
        self.original_forwards = {}
        self.patched_modules = {}
        
    def patch_attention(self, module, name):

        original_forward = module.forward
        self.original_forwards[name] = original_forward
        self.patched_modules[name] = module
        
  
        def patched_forward(self_mod, x: torch.Tensor) -> torch.Tensor:
            B, N, C = x.shape
            qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = module.q_norm(q), module.k_norm(k)

     
            q = q * module.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            
          
            self.attention_maps[name] = attn.detach()
            
            attn = module.attn_drop(attn)
            x_out = attn @ v

            x_out = x_out.transpose(1, 2).reshape(B, N, C)
            x_out = module.proj(x_out)
            x_out = module.proj_drop(x_out)
            return x_out
            

        module.forward = patched_forward.__get__(module, type(module))
        
    def clear(self):
        self.attention_maps.clear()
        
    def remove_hooks(self):
   
        for name, module in self.patched_modules.items():
            if name in self.original_forwards:
                module.forward = self.original_forwards[name].__get__(module, type(module))
        self.original_forwards.clear()
        self.patched_modules.clear()
        self.attention_maps.clear()
