# torchgl
`torchgl` is a simple library for copying data from PyTorch CUDA tensors to 
[ModernGL](https://github.com/moderngl/moderngl) textures directly on the GPU. 

It wraps CUDA graphics interoperability calls, which avoids slow cpu round-trips.

# Use Cases
- Render visualizations in real-time
- Use GLSL shaders in pre- or post-processing
- Combine ML with traditional graphics pipelines

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

This package depends on [CUDA Python](https://nvidia.github.io/cuda-python/latest/) bindings. It is recommended to 
match the version are using with PyTorch, e.g. if you want to use CUDA 12.8
```bash
pip install cuda-python==12.8
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

# Supported Formats
In ModernGL, the texture format ([see docs](https://moderngl.readthedocs.io/en/5.8.2/topics/texture_formats.html)) 
is a combination of an internal OpenGL format and a data type used for supplying data to the texture. 

In `torchgl`, The dtype of the tensor will match the dtype you would use supply pixel data to the texture. 
E.g. `"f1"` textures will convert to and from `unit8` tensors, and `"f2"` textures will convert to and from `float16`. 

For simplicity, we assume the internal format of the texture is expected one. If you override
the ModernGL internal format, or if your OpenGL driver selects a different internal format, conversion may
not behave as expected.

We explicitly exclude the formats `"ni1"` and `"ni2"` because these have unsigned internal formats but are signed
when supplying pixel data, and we feel this will lead to confusion. 

# Advanced Usage
For more control of the resource management and synchronization between CUDA and OpenGL, 
you can allocate your textures ahead of time and register them once for interop using `register()`. Then, use
`map()` and `unmap()` to efficiently pipeline the CUDA and OpenGL and reduce unnecessary synchronization points.

Streams are also supported, although you should be aware of how they work with the PyTorch allocation of CUDA tensors, 
see [docs here](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-semantics).

# Related Packages
[torch2moderngl](https://github.com/geospaitial-lab/torch2moderngl) provides similar basic usage for Textures only.
