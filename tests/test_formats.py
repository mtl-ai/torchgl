import moderngl
import torch

import torchgl as tgl


def test():
    h, w = 1077, 1923
    device = "cuda"

    ctx = moderngl.create_context(standalone=True)
    for k, v in ctx.info.items():
        print(k, v)

    for test_manual_mapping in (False, True):
        for gl_format, torch_dtype in tgl._gl_to_torch_dtype.items():
            for c in (1, 2, 4):
                if torch_dtype.is_floating_point:
                    input_tensor = torch.rand(h, w, c, dtype=torch_dtype, device=device)
                else:
                    info = torch.iinfo(torch_dtype)
                    input_tensor = torch.randint(
                        info.min,
                        info.max + 1,
                        size=(h, w, c),
                        dtype=torch_dtype,
                        device=device,
                    )

                if test_manual_mapping:
                    converted_texture = ctx.texture(
                        size=(w, h), components=c, dtype=gl_format
                    )
                    tgl.register(converted_texture, "w")
                    tgl.map(converted_texture)
                    tgl.to_texture(input_tensor, converted_texture)
                    tgl.unmap(converted_texture)
                    tgl.unregister(converted_texture)
                else:
                    converted_texture = tgl.to_texture(input_tensor)

                buf = bytearray(
                    converted_texture.read()
                )  # need a read/writeable buffer
                output_tensor = torch.frombuffer(buf, dtype=torch_dtype)
                # need to flip because data is read out from bottom left in GL
                output_tensor = output_tensor.reshape(h, w, c).to(device).flip(dims=[0])

                assert output_tensor.dtype == input_tensor.dtype
                assert torch.equal(output_tensor, input_tensor)

                input_texture = ctx.texture(
                    size=(w, h), components=c, dtype=gl_format, data=buf
                )

                if test_manual_mapping:
                    tgl.register(input_texture, "r")
                    tgl.map(input_texture)
                    converted_tensor = tgl.to_tensor(input_texture)
                    tgl.unmap(input_texture)
                    tgl.unregister(input_texture)
                else:
                    converted_tensor = tgl.to_tensor(input_texture)

                assert converted_tensor.dtype == input_tensor.dtype
                assert torch.equal(converted_tensor, input_tensor)
