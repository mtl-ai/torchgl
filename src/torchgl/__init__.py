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
    torch.float32: "f4",
}


def _create_cuda_descriptor(
    c: int, dtype: torch.dtype
) -> tuple[int, int, int, int, cudart.cudaChannelFormatKind]:
    """
    Create the matching cudaChannelFormatDesc for a given channels and dtype of a tensor.

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

    bits: int = (
        torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)
    ).bits

    bits_x, bits_y, bits_z, bits_w = [bits if i < c else 0 for i in range(4)]

    return bits_x, bits_y, bits_z, bits_w, kind


_cuda_descriptor_to_torch = {
    _create_cuda_descriptor(c, dtype): (c, dtype)
    for dtype in _torch_to_gl_dtype.keys()
    for c in (1, 2, 4)
}


def _object_key(
    obj: moderngl.Texture | moderngl.Buffer,
) -> tuple[type[moderngl.Texture | moderngl.Buffer], int]:
    if isinstance(obj, moderngl.Texture):
        t = moderngl.Texture
    elif isinstance(obj, moderngl.Buffer):
        t = moderngl.Buffer
    else:
        raise ValueError("Object is not a Texture or a Buffer")

    return t, obj.glo


_registered_objects: dict[
    tuple[type[moderngl.Texture | moderngl.Buffer], int],
    tuple[cudart.cudaGraphicsResource_t, Mode],
] = dict()


def register(obj: moderngl.Texture | moderngl.Buffer, mode: Mode):
    """
    Register a ModernGL Texture or Buffer for CUDA interoperability.

    This function is provided for advanced usage. For basic usage, the object will automatically be registered
    and unregistered as needed.

    If you manually register an object, you are also responsible for mapping and unmapping the object as required.

    Parameters
    ----------
    obj : moderngl.Texture | moderngl.Buffer
        The ModernGL object to register. For textures, only 1, 2, or 4 components
        are supported.
    mode : {"r", "w", "rw"}
        Access mode describing how PyTorch-CUDA will use the object:
        - "r"  : read-only (usable with `to_tensor()`)
        - "w"  : write-only (usable with `to_texture() or to_buffer()`)
        - "rw" : read–write
    """
    key = _object_key(obj)
    if key in _registered_objects:
        raise ValueError("Object already registered")

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

    if isinstance(obj, moderngl.Texture):
        if obj.components == 3:
            raise ValueError("Textures with 3 components are not supported")

        # see https://raw.githubusercontent.com/KhronosGroup/OpenGL-Registry/refs/heads/main/xml/gl.xml
        _GL_TEXTURE_2D = 0x0DE1

        resource = _check_cuda_error(
            cudart.cudaGraphicsGLRegisterImage(obj.glo, _GL_TEXTURE_2D, flags)
        )

    elif isinstance(obj, moderngl.Buffer):
        resource = _check_cuda_error(
            cudart.cudaGraphicsGLRegisterBuffer(obj.glo, flags)
        )
    else:
        assert False  # unreachable

    _registered_objects[key] = (resource, mode)


def unregister(obj: moderngl.Texture | moderngl.Buffer):
    """
    Unregister a ModernGL Texture or Buffer for CUDA interoperability.

    This function is provided for advanced usage. For basic usage, the buffer will automatically be registered
    and unregistered as needed.

    Parameters
    ----------
    obj : moderngl.Texture | moderngl.Buffer
        The ModernGL object to unregister. Must have been registered previously with `register()`.
    """

    key = _object_key(obj)
    if key not in _registered_objects:
        raise ValueError("Object not registered")

    resource, _ = _registered_objects[key]

    _check_cuda_error(cudart.cudaGraphicsUnregisterResource(resource))

    del _registered_objects[key]


def map(obj: moderngl.Texture | moderngl.Buffer):
    """
    Map a ModernGL Texture or Buffer for CUDA interoperability.

    Mapping will ensure all pending OpenGL operations complete before work begins in the current CUDA stream
    (given by `torch.cuda.current_stream()`).

    This function is provided for advanced usage. For basic usage, the object will automatically be mapped and
    unmapped as required.

    Parameters
    ----------
    obj : moderngl.Texture | moderngl.Buffer
        The object to map. It must have been previously registered using
        `register()`.
    """

    key = _object_key(obj)
    if key not in _registered_objects:
        raise ValueError("Object not registered")

    resource, _ = _registered_objects[key]

    stream = cudart.cudaStream_t(torch.cuda.current_stream().cuda_stream)

    _check_cuda_error(cudart.cudaGraphicsMapResources(1, resource, stream))


def unmap(obj: moderngl.Texture | moderngl.Buffer):
    """
    Unmap a ModernGL Texture or Buffer for CUDA interoperability.

    Unmapping will ensure all work in the current CUDA stream (given by `torch.cuda.current_stream()`) will complete
    before any OpenGL work starts.

    This function is provided for advanced usage. For basic usage, the object will automatically be mapped and
    unmapped as required.

    Parameters
    ----------
    obj : moderngl.Texture
        The texture to unmap. It must have been previously mapped using `map()`.
    """

    key = _object_key(obj)
    if key not in _registered_objects:
        raise ValueError("Object not registered")

    resource, _ = _registered_objects[key]

    stream = cudart.cudaStream_t(torch.cuda.current_stream().cuda_stream)

    _check_cuda_error(cudart.cudaGraphicsUnmapResources(1, resource, stream))


def to_tensor(
    obj: moderngl.Texture | moderngl.Buffer, tensor: torch.Tensor = None
) -> torch.Tensor:
    """
    Copy a ModernGL Texture or Buffer to a CUDA Tensor.

    In the case of a Texture object, the returned Tensor will have shape (H, W, C), where
    (W, H) is the size of the texture and C is the number of components (1, 2, or 4).
    An optional output tensor can be provided, but it must match the expected shape and dtype. Note that with the
    current implementation, there is still an intermediate tensor created.

    In the case of a Buffer object, the returned Tensor will have shape (N) where N is the Buffer size in bytes,
    and dtype uint8. If an output Tensor is provided, it must be contiguous and have the same size as
    the buffer in bytes, but may have any shape or dtype. In this case, no intermediate Tensors will need to be
    created.

    If the texture is not registered, it will temporarily be registered and mapped for the copy.

    Parameters
    ----------
    obj : moderngl.Texture | moderngl.Buffer
        The ModernGL object to read from.

    tensor: optional output Tensor

    Returns
    -------
    torch.Tensor
        A CUDA tensor containing the texture's pixel data.
    """

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    stream = cudart.cudaStream_t(torch.cuda.current_stream().cuda_stream)

    if tensor is not None:
        if not tensor.is_cuda:
            raise ValueError("Tensor must be on CUDA device")

        if not tensor.is_contiguous():
            raise ValueError("Tensor must be contiguous")

    key = _object_key(obj)
    is_registered_by_user = key in _registered_objects
    if not is_registered_by_user:
        register(obj, "r")
        map(obj)

    resource, mode = _registered_objects[key]
    if mode not in ("r", "rw"):
        raise ValueError(f"Invalid access mode '{mode}' (need 'r' or 'rw')")

    if isinstance(obj, moderngl.Texture):
        array = _check_cuda_error(
            cudart.cudaGraphicsSubResourceGetMappedArray(resource, 0, 0)
        )

        desc, extent, _flags = _check_cuda_error(cudart.cudaArrayGetInfo(array))

        descriptor = (desc.x, desc.y, desc.z, desc.w, desc.f)
        assert (
            descriptor in _cuda_descriptor_to_torch
        )  # assume we have covered every possibility
        c, dtype = _cuda_descriptor_to_torch[descriptor]
        w, h = extent.width, extent.height
        assert ((w, h), c) == (
            obj.size,
            obj.components,
        )  # assume that the ModernGL attributes match the underlying texture

        _tensor = torch.empty((h, w, c), dtype=dtype, device=device)
        b = _tensor.dtype.itemsize

        _check_cuda_error(
            cudart.cudaMemcpy2DFromArrayAsync(
                _tensor.data_ptr(),
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
        _tensor = _tensor.flip(dims=[0]).contiguous()

        if tensor is not None:
            if tensor.shape != _tensor.shape:
                raise ValueError(
                    f"Expected tensor to have shape {tuple(_tensor.shape)}, but found {(tuple(tensor.shape))}"
                )
            if tensor.dtype != _tensor.dtype:
                raise ValueError(
                    f"Expected tensor to have dtype {_tensor.dtype}, but found {tensor.dtype}"
                )
            tensor.copy_(_tensor)
        else:
            tensor = _tensor

    elif isinstance(obj, moderngl.Buffer):
        ptr, size = _check_cuda_error(
            cudart.cudaGraphicsResourceGetMappedPointer(resource)
        )
        assert size == obj.size

        if tensor is None:
            tensor = torch.empty(size=size, device=device, dtype=torch.uint8)

        tensor_size = tensor.nelement() * tensor.element_size()

        if tensor_size != size:
            raise ValueError(
                f"Buffer size {size} and tensor size {tensor_size} in bytes don't match"
            )

        _check_cuda_error(
            cudart.cudaMemcpyAsync(
                tensor.data_ptr(),
                ptr,
                size,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                stream,
            )
        )

    else:
        assert False  # unreachable

    if not is_registered_by_user:
        unmap(obj)
        unregister(obj)

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
        raise ValueError("Tensor must be on CUDA device")

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

    if (texture.size, texture.components) != ((w, h), c):
        raise ValueError(
            f"Texture with size {texture.size} and components {texture.components} does not correspond to tensor with shape {tuple(tensor.shape)}"
        )

    key = _object_key(texture)
    is_already_registered = key in _registered_objects
    if not is_already_registered:
        register(texture, "w")
        map(texture)

    resource, mode = _registered_objects[key]

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


def to_buffer(tensor: torch.Tensor, buffer: moderngl.Buffer = None) -> moderngl.Buffer:
    """


    asdfasdfasdfasdfasdfadsfadfadfadsfasdf
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
        raise ValueError("Tensor must be on CUDA device")

    tensor = tensor.flatten().contiguous()
    tensor_size = tensor.nelement() * tensor.element_size()

    if buffer is None:
        ctx = moderngl.get_context()
        buffer = ctx.buffer(reserve=tensor_size)

    if tensor_size != buffer.size:
        raise ValueError(
            f"Tensor has size {tensor_size} in bytes, but buffer has size {buffer.size}"
        )

    key = _object_key(buffer)
    is_already_registered = key in _registered_objects
    if not is_already_registered:
        register(buffer, "w")
        map(buffer)

    resource, mode = _registered_objects[key]

    if mode not in ("w", "rw"):
        raise ValueError(f"Invalid texture access mode '{mode}' (need 'w' or 'rw')")

    ptr, size = _check_cuda_error(cudart.cudaGraphicsResourceGetMappedPointer(resource))
    assert size == buffer.size

    stream = cudart.cudaStream_t(torch.cuda.current_stream().cuda_stream)

    _check_cuda_error(
        cudart.cudaMemcpyAsync(
            ptr,
            tensor.data_ptr(),
            size,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            stream,
        )
    )

    if not is_already_registered:
        unmap(buffer)
        unregister(buffer)

    return buffer
