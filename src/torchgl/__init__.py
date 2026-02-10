from typing import Any, Literal

import cuda.bindings.runtime as cudart
import moderngl
import torch


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

# Mapping of supported ModernGL texture formats to their corresponding PyTorch dtypes.
# The PyTorch dtype reflects the NumPy dtype that should be used when supplying texture data.
#
# The signed normalized formats 'ni1' and 'ni2' are intentionally excluded due to their
# potential for user confusion.
#
# Note: These conversions assume that ModernGL (and the underlying OpenGL driver) selects the
# expected internal texture format. If the user overrides the internal format, or if the driver
# chooses a different one, the dtype mapping may no longer be valid.
_gl_to_torch_dtype: dict[str, torch.dtype] = {
    "f1": torch.uint8,
    "f2": torch.float16,
    "f4": torch.float32,
    "u1": torch.uint8,
    "u2": torch.uint16,
    "u4": torch.uint32,
    "i1": torch.int8,
    "i2": torch.int16,
    "i4": torch.int32,
    "nu1": torch.uint8,
    "nu2": torch.uint16,
}
# Builds the reverse mapping: torch.dtype -> list[str].
# Multiple ModernGL formats may correspond to the same torch dtype.
_torch_to_gl_formats: dict[torch.dtype, list[str]] = {}
for _gl_format, _torch_dtype in _gl_to_torch_dtype.items():
    _torch_to_gl_formats.setdefault(_torch_dtype, []).append(_gl_format)

# glo -> (resource handle, mode)
_registered_textures: dict[int, tuple[cudart.cudaGraphicsResource_t, Mode]] = dict()


def register(texture: moderngl.Texture, mode: Mode):
    """
    Registers a texture for interop with torch/cuda.

    After registering, the texture can be mapped and converted to a tensor.

    Cannot register the same texture again until it is unregistered.

    The mode parameter determines how the texture will be accessed by torch/cuda
    "r": will only be read from by torch/cuda (can be used in to_tensor())
    "w": will only be written to by torch/cuda (can be used in to_texture())
    "rw": will be both read from and written to by torch/cuda

    """

    if texture.glo in _registered_textures:
        raise ValueError("Texture already registered")

    if texture.components == 3:
        raise ValueError("Textures with 3 components not supported")

    if texture.dtype not in _gl_to_torch_dtype:
        raise ValueError(
            f"Texture format {texture.dtype} is not one of {list(_gl_to_torch_dtype.keys())}"
        )

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
    Unregisters a texture.

    The texture can no longer be mapped and converted to a tensor.
    """
    if texture.glo not in _registered_textures:
        raise ValueError("Texture not registered")

    resource, _ = _registered_textures[texture.glo]

    _check_cuda_error(cudart.cudaGraphicsUnregisterResource(resource))

    del _registered_textures[texture.glo]


def map(texture: moderngl.Texture):
    """
    Maps a texture so that it can be converted to or from a tensor.

    All OpenGL work will complete before work begins in thh current torch/cuda stream (torch.cuda.current_stream())
    """

    if texture.glo not in _registered_textures:
        raise ValueError("Texture not registered")
    resource, _ = _registered_textures[texture.glo]

    stream = cudart.cudaStream_t(torch.cuda.current_stream().cuda_stream)

    _check_cuda_error(cudart.cudaGraphicsMapResources(1, resource, stream))


def unmap(texture: moderngl.Texture):
    """
    Unmaps a texture so it can no longer be converted to or from a tensor.

    All work in current torch/cuda stream (torch.cuda.current_stream()) will complete before subsequent OpenGL work begins.
    """

    if texture.glo not in _registered_textures:
        raise ValueError("Texture not registered.")
    resource, _ = _registered_textures[texture.glo]

    stream = cudart.cudaStream_t(torch.cuda.current_stream().cuda_stream)

    _check_cuda_error(cudart.cudaGraphicsUnmapResources(1, resource, stream))
    pass


def to_tensor(texture: moderngl.Texture) -> torch.Tensor:
    """
    Convert a ModernGL texture into a CUDA-backed PyTorch tensor.

    The returned tensor resides on the current CUDA device and has a dtype
    that is inferred from the texture's format. The shape of the tensor is

    Parameters
    ----------
    texture : moderngl.Texture
        The ModernGL texture to read from.

    Returns
    -------
    torch.Tensor
        A CUDA tensor containing the texture's pixel data.

    Raises
    ------
    ValueError
        If the texture is registered with an access mode that does not permit
        reading (must be 'r' or 'rw').
    RuntimeError
        If any CUDA runtime call fails during mapping or data transfer.

    Notes
    -----
    If the texture was not previously registered by the user, it will be
    automatically registered, mapped, unmapped, and unregistered as part of
    this call.
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

    w, h = texture.size
    c = texture.components
    assert texture.dtype in _gl_to_torch_dtype
    dtype = _gl_to_torch_dtype[texture.dtype]
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

    if tensor.ndim != 3:
        raise ValueError("Tensor must have 3 dims")

    h, w, c = tensor.shape
    if c not in (1, 2, 4):
        raise ValueError(
            f"Only tensors with 1, 2, or 4 channels are supported, got {c}"
        )

    if tensor.dtype not in _torch_to_gl_formats:
        raise ValueError(f"Tensor dtype must be in {list(_torch_to_gl_formats.keys())}")
    b = tensor.dtype.itemsize

    expected_formats = _torch_to_gl_formats[tensor.dtype]
    if texture is None:
        ctx = moderngl.get_context()
        texture = ctx.texture(
            (w, h), components=c, dtype=expected_formats[0]
        )  # assume the first format is preferred

    if texture.dtype not in expected_formats:
        raise ValueError(
            f"Expected a texture with format one of {expected_formats}, got {texture.dtype}"
        )

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
