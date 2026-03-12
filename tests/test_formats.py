import moderngl
import torch

import torchgl as tgl


def test_formats():
    h, w = 1077, 1923
    device = "cuda"

    ctx = moderngl.create_context(standalone=True)
    for k, v in ctx.info.items():
        print(k, v)

    # these give expected results
    # ni1, and ni2 are surprising
    nice_gl_to_torch_types = {
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

    for test_manual_mapping in (False, True):
        for gl_format, torch_dtype in nice_gl_to_torch_types.items():
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

if __name__ == "__main__":
    test_formats()