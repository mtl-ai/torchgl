# torchgl
`torchgl` is a simple library for sharing data between PyTorch and [ModernGL](https://github.com/moderngl/moderngl) 
directly on the GPU. 

It wraps CUDA graphics interoperability calls, which avoids slow CPU round-trips. No C++ compiling is needed
since we use the python CUDA API. 

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
## Other examples
See [other examples](examples) here.

# Installation
`torchgl` is can be installed with pip

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

# Texture Formats
`torchgl` supports 1, 2, or 4 component Textures (3 component textures are not supported by CUDA). 
For most ModernGL dtypes, the converted Tensor type matches what you would use to supply pixel data.

| ModernGL dtype | PyTorch dtype  |
|----------------|----------------|
| f1             | uint8          |
| f2             | half           |        
| f4             | float32        |       
| u1             | uint8          |      
| u2             | uint16         |      
| u4             | uint32         |    
| i1             | int8           |   
| i2             | int16          |  
| i4             | in32           |

The exceptions are "ni1", and "ni2" which are really unsigned internally, but the pixel data
is expected to be signed (GL_BYTE or GL_SHORT), see https://github.com/moderngl/moderngl/blob/main/src/moderngl.cpp#L800. 
We suggest avoiding these types unless you know really know what you are doing. 

Also, if you override ModernGL internal format conversion may not behave as expected.


# Advanced Usage
For more control of the resource management and synchronization between CUDA and OpenGL, 
you can allocate your textures and buffers ahead of time and register them once for interop using `register()`. Then, use
`map()` and `unmap()` to efficiently pipeline the CUDA and OpenGL and reduce unnecessary synchronization points.

Streams are also supported, although you should be aware of how they work with the PyTorch allocation of CUDA tensors, 
see [docs here](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-semantics).

# Related Packages
[torch2moderngl](https://github.com/geospaitial-lab/torch2moderngl) provides similar basic usage for Textures only.
