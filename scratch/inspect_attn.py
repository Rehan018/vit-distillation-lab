import timm
import inspect

model = timm.create_model('vit_tiny_patch16_224')
attn_module = model.blocks[0].attn
print("Type:", type(attn_module))
print("\nSource code:")
print(inspect.getsource(type(attn_module)))
