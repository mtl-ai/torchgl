import moderngl

import torch
import moderngl_window as mglw

import torchgl as tgl


class Particles(mglw.WindowConfig):
    title = "Particles"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        n = 100
        d = 2
        device = "cuda"
        self.k = 1.0 / n
        self.x = torch.rand(n, 2, device=device) * 2.0 - 1.0
        self.v = torch.zeros(n, 2, device=device)

        vtx_shader = """
            #version 330

            in vec2 in_pos; // 2D position in [-1, 1]
            uniform float point_size;

            void main() {
                gl_Position = vec4(in_pos, 0.5, 1.0);
                gl_PointSize = point_size;
            }
        """

        frag_shader = """
            #version 330

            out vec4 fragColor;

            void main() {
                fragColor = vec4(1.0, 0.8, 0.2, 1.0);
            }
        """

        prog = self.ctx.program(vertex_shader=vtx_shader, fragment_shader=frag_shader)
        self.vbo = self.ctx.buffer(reserve=n * d * 4)
        self.vao = self.ctx.vertex_array(prog, [(self.vbo, "2f", "in_pos")])
        self.vao.program["point_size"] = 5.0
        self.vao.mode = moderngl.POINTS
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

        self.counter = 0

    def on_render(self, time: float, frametime: float):
        with torch.inference_mode():
            x = self.x
            v = self.v
            k = self.k

            r = x[None, :, :] - x[:, None, :]  # (N, N, d)
            r2 = r.square().sum(dim=-1, keepdim=True)  # (N, N, 1)

            F = k * r / (r2 + 1e-4)  # (N, N, d)
            F.diagonal().fill_(0.0)  # set self-interaction to zero

            total_F = torch.sum(F, dim=1) / 2  # (N, d)

            dt = frametime
            x = x + dt * v
            v = v + dt * total_F

            self.x = x
            self.v = v

            tgl.to_buffer(self.x, self.vbo)

        self.ctx.clear()
        self.vao.render()


if __name__ == "__main__":
    Particles.run()
