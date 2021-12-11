from numpy.core.fromnumeric import shape
import taichi as ti
import numpy as np

lin_iters = 20

N = 64
dt = 0.1
diff = 0.0
visc = 0.0
force = 5e5
source = 100.0

dvel = False

v = ti.Vector.field(2, float, shape=(N + 2, N + 2), offset = (-1, -1))
v_prev = ti.Vector.field(2, float, shape=(N + 2, N + 2), offset = (-1, -1))
dens = ti.field(float, shape=(N + 2, N + 2), offset = (-1, -1))
dens_prev = ti.field(float, shape=(N + 2, N + 2), offset = (-1, -1))

div = ti.field(float, shape=(N + 2, N + 2), offset = (-1, -1))
p = ti.field(float, shape=(N + 2, N + 2), offset = (-1, -1))
pixels = ti.field(float, shape=(N, N))

@ti.kernel
def add_source(a : ti.template(), b : ti.template()):
    for i, j in a:
        a[i, j] += dt * b[i, j]

@ti.kernel
def swap(a : ti.template(), b : ti.template()):
    for i, j in a:
        a[i, j], b[i, j] = b[i, j], a[i, j]

@ti.func
def set_bnd(x : ti.template()):
    for i in range(N):
        x[-1, i] = x[0, i]
        x[N, i] = x[N - 1, i]
        x[i, -1] = x[i, 0]
        x[i, N] = x[i, N - 1]
    x[-1, -1] = (x[0, -1] + x[-1, 0]) / 2.0
    x[-1, N] = (x[0, N] + x[-1, N - 1]) / 2.0
    x[N, -1] = (x[N - 1, -1] + x[N, 0]) / 2.0
    x[N, N] = (x[N - 1, N] + x[N, N - 1]) / 2.0

@ti.kernel
def lin_solve(x : ti.template(), x0 : ti.template(), a : float, c : float):
    for i, j in ti.ndrange(N, N):
        x[i, j] = (x0[i, j] + a * (x[i - 1, j] + x[i + 1, j] + x[i, j - 1] + x[i, j + 1])) / c
    set_bnd(x)

def diffuse(a, a_prev, diff):
    k = dt * diff * N * N
    for t in range(lin_iters):
        lin_solve(a, a_prev, k, 1.0 + 4.0 * k)

@ti.kernel
def advect(d : ti.template(), d0 : ti.template(), v : ti.template() ):
    dt0 = dt * N
    for i, j in ti.ndrange(N, N):
        x, y = i - dt0 * v[i, j][0], j - dt0 * v[i, j][1]
        if (x < 0.5): x = 0.5
        if (x > N + 0.5): x = N + 0.5
        i0, i1 = int(x), int(x) + 1
        if (y < 0.5): y = 0.5
        if (y > N + 0.5): y = N + 0.5
        j0, j1 = int(y), int(y) + 1
        s1, s0, t1, t0 = x - i0, i1 - x, y - j0, j1 - y
        d[i, j] = s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) + s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])
    set_bnd(d)

def fft_project(v):
    u0 = np.zeros(shape = (N + 2, N))
    v0 = np.zeros(shape = (N + 2, N))
    for i, j in ti.ndrange(N, N):
        u0[i, j], v0[i, j] = v[i, j][0], v[i, j][1]
    u0 = np.fft.fft2(u0)
    v0 = np.fft.fft2(v0)
    for i, j in ti.ndrange(N + 2, N):
        x, y = i, j
        if j > N // 2 : j = j - N
        r = x * x + y * y
        if (r == 0.0): continue
        f = ti.exp(-r*dt*visc)
        U, V = u0[i,j], v0[i,j]
        u0[i, j] = f * np.complex((1-x*x/r)*U.real+(-x*y/r)*V.real, (1-x*x/r)*U.imag+(-x*y/r)*V.imag)
        v0[i, j] = f * np.complex((-y*x/r)*U.real+(1-y*y/r)*V.real,(-y*x/r)*U.imag+(1-y*y/r)*V.imag)
    u0 = np.fft.ifft2(u0)
    v0 = np.fft.ifft2(v0)
    f = 1.0/(N*N)
    for i, j in ti.ndrange(N, N):
        v[i, j][0], v[i, j][1] = f * u0[i, j], f * v0[i, j]
    print("Okay")

def dens_step():
    add_source(dens, dens_prev)
    swap(dens, dens_prev)
    diffuse(dens, dens_prev, diff)
    swap(dens, dens_prev)
    advect(dens, dens_prev, v)

def vel_step():
    add_source(v, v_prev)
    swap(v, v_prev)
    diffuse(v, v_prev, visc)
    fft_project(v)
    swap(v, v_prev)
    advect(v, v_prev, v_prev)
    fft_project(v)