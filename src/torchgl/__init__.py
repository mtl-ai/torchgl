from typing import Any, Literal

import cuda.bindings.runtime as cudart
import moderngl
import torch

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch with CUDA backend is required")


def _check_cuda_error(result: Any):
    # cuda calls should return a tuple with the error has the first item
    assert isinstance(result, tuple)
    err: cudart.cudaError_t = result[0]
    if err != 0:
        _, msg = cudart.cudaGetErrorString(err)
        raise RuntimeError(f"CUDA error: {msg.decode('ascii')}")
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


Mode = Literal["r", "w", "rw"]

# preferred ModernGL dtype for a given torch tensor dtype
_torch_to_gl_dtype = {
    torch.uint8: "f1",
    torch.uint16: "u2",
    torch.uint32: "u4",
    torch.int8: "i1",
    torch.int16: "i2",
    torch.int32: "i4",
    torch.float16: "f2",
    torch.float32: "f4"
}

def _create_descriptor(c: int, dtype: torch.dtype) -> tuple[int, int, int, int, cudart.cudaChannelFormatKind]:
    """
    Match the information in cudaChannelFormatDesc to the info needed to create a tensor.

    Returns bit-depth in (x, y, z, w) components + kind (either signed, unsigned, or float).
    """

    if c not in (1, 2, 4):
        raise ValueError(f"channels must be 1, 2, or 4, got {c}")

    if dtype.is_floating_point:
        kind = cudart.cudaChannelFormatKind.cudaChannelFormatKindFloat
    elif dtype.is_signed:
        kind = cudart.cudaChannelFormatKind.cudaChannelFormatKindSigned
    else:
        kind = cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsigned

    bits: int = (torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)).bits

    bits_x, bits_y, bits_z, bits_w = [bits if i < c else 0 for i in range(4)]

    return bits_x, bits_y, bits_z, bits_w, kind

_descriptor_to_torch_channels_and_dtype = {
    _create_descriptor(c, dtype) : (c, dtype)
    for dtype in _torch_to_gl_dtype.keys() for c in (1, 2, 4)
}


# glo -> (resource handle, mode)
_registered_textures: dict[int, tuple[cudart.cudaGraphicsResource_t, Mode]] = dict()


def register(texture: moderngl.Texture, mode: Mode):
    """
    Register a ModernGL texture for CUDA interoperability.

    This function is provided for advanced usage. For basic usage, the texture will automatically be registered
    and unregistered as needed.

    If you manually register a texture, you are also responsible for mapping or unmapping the texture as required.

    Parameters
    ----------
    texture : moderngl.Texture
        The ModernGL texture to register. Only textures with 1, 2, or 4 components
        are supported. The texture formats "ni1" and "ni2" are explicitly not supported.
    mode : {"r", "w", "rw"}
        Access mode describing how PyTorch-CUDA will use the texture:
        - "r"  : read-only (usable with `to_tensor()`)
        - "w"  : write-only (usable with `to_texture()`)
        - "rw" : read–write
    """

    if texture.glo in _registered_textures:
        raise ValueError("Texture already registered")

    if texture.components == 3:
        raise ValueError("Textures with 3 components not supported")

    if mode not in (
        "r",
        "w",
        "rw",
    ):
        raise ValueError(f"Mode {mode} is not one of 'r', 'w', 'rw'")

    flags = {
        "r": cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly,
        "w": cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
        "rw": cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsNone,
    }[mode]

    # see https://raw.githubusercontent.com/KhronosGroup/OpenGL-Registry/refs/heads/main/xml/gl.xml
    _GL_TEXTURE_2D = 0x0DE1

    resource = _check_cuda_error(
        cudart.cudaGraphicsGLRegisterImage(texture.glo, _GL_TEXTURE_2D, flags)
    )

    _registered_textures[texture.glo] = (resource, mode)


def unregister(texture: moderngl.Texture):
    """
    Unregister a ModernGL texture for CUDA interoperability.

    This function is provided for advanced usage. For basic usage, the texture will automatically be registered
    and unregistered as needed.

    Parameters
    ----------
    texture : moderngl.Texture
        The ModernGL texture to unregister. Must have been registered previously with `register()`.
    """
    if texture.glo not in _registered_textures:
        raise ValueError("Texture not registered")

    resource, _ = _registered_textures[texture.glo]

    _check_cuda_error(cudart.cudaGraphicsUnregisterResource(resource))

    del _registered_textures[texture.glo]


def map(texture: moderngl.Texture):
    """
    Map a ModernGL texture for CUDA interoperability.

    Mapping will ensure all pending OpenGL operations complete before work begins in the current CUDA stream
    (given by `torch.cuda.current_stream()`).

    This function is provided for advanced usage. For basic usage, the texture will automatically be mapped and
    unmapped as required.

    Parameters
    ----------
    texture : moderngl.Texture
        The texture to map. It must have been previously registered using
        `register()`.
    """

    if texture.glo not in _registered_textures:
        raise ValueError("Texture not registered")
    resource, _ = _registered_textures[texture.glo]

    stream = cudart.cudaStream_t(torch.cuda.current_stream().cuda_stream)

    _check_cuda_error(cudart.cudaGraphicsMapResources(1, resource, stream))


def unmap(texture: moderngl.Texture):
    """
    Unmap a ModernGL texture for CUDA interoperability.

    Unmapping will ensure all work in the current CUDA stream (given by `torch.cuda.current_stream()`) will complete
    before any OpenGL work starts.

    This function is provided for advanced usage. For basic usage, the texture will automatically be mapped and
    unmapped as required.

    Parameters
    ----------
    texture : moderngl.Texture
        The texture to unmap. It must have been previously mapped using `map()`.
    """

    if texture.glo not in _registered_textures:
        raise ValueError("Texture not registered.")
    resource, _ = _registered_textures[texture.glo]

    stream = cudart.cudaStream_t(torch.cuda.current_stream().cuda_stream)

    _check_cuda_error(cudart.cudaGraphicsUnmapResources(1, resource, stream))
    pass


def to_tensor(texture: moderngl.Texture) -> torch.Tensor:
    """
    Copy a ModernGL texture to a CUDA tensor.

    The returned tensor resides on the current device and has a dtype
    that is inferred from the texture's format. The returned tensor will have shape (H, W, C), where
    (W, H) is the size of the texture and C is the number of components (1, 2, or 4).

    If the texture is not registered, it will temporarily be registered and mapped for the copy.

    Parameters
    ----------
    texture : moderngl.Texture
        The ModernGL texture to read from.

    Returns
    -------
    torch.Tensor
        A CUDA tensor containing the texture's pixel data.
    """

    is_registered_by_user = texture.glo in _registered_textures
    if not is_registered_by_user:
        register(texture, "r")
        map(texture)

    resource, mode = _registered_textures[texture.glo]
    if mode not in ("r", "rw"):
        raise ValueError(f"Invalid texture access mode '{mode}' (need 'r' or 'rw')")

    array = _check_cuda_error(
        cudart.cudaGraphicsSubResourceGetMappedArray(resource, 0, 0)
    )

    desc, extent, _flags = _check_cuda_error(cudart.cudaArrayGetInfo(array))

    descriptor = (desc.x, desc.y, desc.z, desc.w, desc.f)
    assert descriptor in _descriptor_to_torch_channels_and_dtype
    c, dtype = _descriptor_to_torch_channels_and_dtype[descriptor]
    w, h = extent.width, extent.height
    assert ((w, h), c) == (texture.size, texture.components)

    device = f"cuda:{torch.cuda.current_device()}"
    tensor = torch.empty((h, w, c), dtype=dtype, device=device)
    b = tensor.dtype.itemsize

    stream = cudart.cudaStream_t(torch.cuda.current_stream().cuda_stream)

    _check_cuda_error(
        cudart.cudaMemcpy2DFromArrayAsync(
            tensor.data_ptr(),
            w * c * b,
            array,
            0,
            0,
            w * c * b,
            h,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            stream,
        )
    )

    tensor = tensor.flip(dims=[0]).contiguous()

    if not is_registered_by_user:
        unmap(texture)
        unregister(texture)

    return tensor


def to_texture(
    tensor: torch.Tensor, texture: moderngl.Texture = None
) -> moderngl.Texture:
    """
    Copy a CUDA tensor into a ModernGL texture.

    If no texture is provided, a new one is created (from the current context) with dimensions and format
    inferred from the tensor. The tensor must have shape (H, W, C) with
    1, 2, or 4 channels, and its dtype must correspond with the ModernGL format of the texture.

    If the texture is not registered, it will temporarily be registered and mapped for the copy.

    Parameters
    ----------
    tensor: torch.Tensor
        A CUDA tensor containing the pixel data.

    texture : moderngl.Texture, optional
        The ModernGL texture to store the pixel data in.

    Returns
    -------
    torch.Tensor
        A CUDA tensor containing the texture's pixel data.
    """

    if not tensor.is_cuda:
        raise ValueError("Tensor must be on cuda device")


    if tensor.ndim != 3:
        raise ValueError("Tensor must have 3 dims")

    h, w, c = tensor.shape
    if c not in (1, 2, 4):
        raise ValueError(
            f"Only tensors with 1, 2, or 4 channels are supported, got {c}"
        )

    if tensor.dtype not in _torch_to_gl_dtype:
         raise ValueError(f"Tensor dtype must be in {list(_torch_to_gl_dtype.keys())}")
    b = tensor.dtype.itemsize

    if texture is None:
        ctx = moderngl.get_context()
        texture = ctx.texture(
            (w, h), components=c, dtype=_torch_to_gl_dtype[tensor.dtype]
        )  # assume the first format is preferred

    is_already_registered = texture.glo in _registered_textures
    if not is_already_registered:
        register(texture, "w")
        map(texture)

    resource, mode = _registered_textures[texture.glo]

    if mode not in ("w", "rw"):
        raise ValueError(f"Invalid texture access mode '{mode}' (need 'w' or 'rw')")

    array = _check_cuda_error(
        cudart.cudaGraphicsSubResourceGetMappedArray(resource, 0, 0)
    )

    stream = cudart.cudaStream_t(torch.cuda.current_stream().cuda_stream)

    tensor = tensor.flip(dims=[0]).contiguous()
    _check_cuda_error(
        cudart.cudaMemcpy2DToArrayAsync(
            array,
            0,
            0,
            tensor.data_ptr(),
            w * c * b,
            w * c * b,
            h,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            stream,
        )
    )

    if not is_already_registered:
        unmap(texture)
        unregister(texture)

    return texture
