# torchgl
`torchgl` is a lightweight library for data transfer between PyTorch-CUDA tensors and 
[ModernGL](https://github.com/moderngl/moderngl) textures. It wraps CUDA–OpenGL interoperability 
calls to enable direct GPU‑to‑GPU movement of data, avoiding slow CPU round‑trips 
and making pipelines more efficient.

# Basic Usage
## Write a tensor into a texture
```python
import moderngl
import torch
import torchgl

# need a moderngl context, e.g.
moderngl.create_context(standalone=True)

# float16
tensor = torch.rand(1080, 1920, 4, device="cuda", dtype=torch.float16)
texture = torchgl.to_texture(tensor)
print(texture.size, texture.components, texture.dtype)  # (1920, 1080) 4 f2

# uint8 with 2 channels
tensor = (255 * tensor[:, :, :2]).to(torch.uint8)
texture = torchgl.to_texture(tensor)
print(texture.size, texture.components, texture.dtype)  # (1920, 1080) 2 f1
```
## Read a texture into a tensor
```python
import moderngl
import torchgl

ctx = moderngl.create_context(standalone=True)
texture = ctx.texture((1920, 1080), 1, dtype="f1")
tensor = torchgl.to_tensor(texture)
print(tensor.shape, tensor.dtype)  # torch.Size([1080, 1920, 1]) torch.uint8
```

# Installation
`torchgl` is (will be) published on PyPI and can be installed with pip

```commandline
pip install torchgl
```
Before installing, make sure you have PyTorch with CUDA support.

# ModernGL Formats
The dtype of the tensor matches the dtype you would use to specify the pixel data as described in 
the [ModernGL Texture Formats](https://moderngl.readthedocs.io/en/5.8.2/topics/texture_formats.html) documentation.
For simplicity, we assume the internal format of the texture is expected one. If you override
the ModernGL internal format, or if your OpenGL driver selects a different internal format, conversion may
not behave as expected.

We explicitly exclude the formats `"ni1"` and `"ni2"` because theses formats behave in a more subtle way (ModernGL uses
unsigned internal format, but transfers the data with a signed type, leading to pixel values which may be surprising).

# Advanced Usage
For more fine-grained resource management and control of the synchronization between CUDA and OpenGL, 
you can use `register()`/`unregister()` functions to register your textures once ahead of time, and use
`map()`/`unmap()` more strategically.
