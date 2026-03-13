import torch
import moderngl_window as mglw

import torchgl as tgl


class WhiteNoise(mglw.WindowConfig):
    title = "Whitenoise"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fbo = self.ctx.framebuffer(self.ctx.texture(self.wnd.size, 4))

    def on_render(self, time: float, frametime: float):

        with torch.inference_mode():
            x = torch.rand(self.fbo.size[1], self.fbo.size[0], 4, device="cuda")
            x = (x * 255).to(torch.uint8)
            tgl.to_texture(x, self.fbo.color_attachments[0])

        self.ctx.copy_framebuffer(dst=self.ctx.fbo, src=self.fbo)


if __name__ == "__main__":
    WhiteNoise.run()
