from phi.jax.flow import *
import matplotlib.pyplot as plt

domain = Box(x=10, y=10)
settings = batch(setting=3)
inflow_rate = tensor([0.1, 0.2, 0.3], settings)
inflow_x = tensor([4, 5, 6], settings)
obstacle_x = wrap([1, 5, 7], settings)  # this affects the pressure matrix -> use NumPy

obstacle = Cuboid(vec(x=obstacle_x, y=6), half_size=vec(x=2, y=1))
inflow = Sphere(x=inflow_x, y=2, radius=2)
plot(obstacle, inflow, overlay="args")


@jit_compile
def step(v, s, p, dt=1.0):
    s = advect.mac_cormack(s, v, dt) + inflow_rate * resample(inflow, to=s, soft=True)
    buoyancy = resample(s * (0, 0.1), to=v)
    v = advect.semi_lagrangian(v, v, dt) + buoyancy * dt
    v, p = fluid.make_incompressible(v, obstacle, Solve(x0=p))
    return v, s, p


v0 = StaggeredGrid(0, 0, domain, x=10, y=10)
smoke0 = CenteredGrid(0, ZERO_GRADIENT, domain, x=20, y=20)
v_trj, s_trj, p_trj = iterate(step, batch(time=100), v0, smoke0, None)


plot(obstacle, inflow, s_trj, animate="time", overlay="args")

plt.show()
