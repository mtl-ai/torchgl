import moderngl
import numpy as np
import torch

import torchgl as tgl


def test_buffers():
    device = "cuda"

    ctx = moderngl.create_context(standalone=True)
    for k, v in ctx.info.items():
        print(k, v)

    x = torch.arange(1000, dtype=torch.uint8, device=device)
    buffer = tgl.to_buffer(x)

    assert x.numpy(force=True).tobytes() == buffer.read()

    buffer = tgl.to_buffer(x[10:15])
    assert buffer.read() == bytes(range(10, 15))

    x = torch.rand(10, 3, device=device, dtype=torch.float16)
    buffer = ctx.buffer(reserve=10 * 3 * 2)
    tgl.to_buffer(x, buffer)
    assert np.all(
        x.numpy(force=True)
        == np.frombuffer(buffer.read(), dtype=np.float16).reshape(10, 3)
    )


if __name__ == "__main__":
    test_buffers()
