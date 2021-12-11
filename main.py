import taichi as ti
ti.init(arch=ti.gpu)
from bruteforce_fluid_solver import *
from taichi_logo_list import taichi_logo

@ti.kernel
def copy2pixels():
    for i, j in pixels:
        pixels[i, j] = dens[i, j]

@ti.kernel
def init_prev():
    for i, j in v_prev:
        v_prev[i, j][0] = v_prev[i, j][1] = dens_prev[i, j] = 0.0

arrs_N = int(N / 8)
arrs = ti.field(float, shape = (arrs_N, arrs_N, 2))
@ti.kernel
def gen_arrs():
    for i, j in ti.ndrange(arrs_N, arrs_N):
        arrs[i, j, 0], arrs[i, j, 1] = v[i * 8, j * 8][0], v[i * 8, j * 8][1]

gui = ti.GUI("Simple Fluid Solver", res = (N, N))
omx, omy, x, y = 0.0, 0.0, 0.0, 0.0
mouse_state = 0
is_recording = False
result_dir = "./output"
video_manager = ti.VideoManager(output_dir = result_dir, framerate = 20)
while gui.running:
    init_prev()
    if(gui.is_pressed(ti.GUI.LMB)):
        x, y = gui.get_cursor_pos()
        mouse_state = 1
    if(gui.is_pressed(ti.GUI.RMB)):
        x, y = gui.get_cursor_pos()
        mouse_state = 2
        omx, omy = x, y
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == 'v':
            dvel = not dvel
        elif gui.event.key == 't':
            for i in taichi_logo:
                dens_prev[i[1], N - i[0]] = source / 10.0
        elif gui.event.key == ti.GUI.ESCAPE:
            gui.running = False
        elif gui.event.key == 's':
            print("Screenshot")
            gui.set_image(pixels)
            gui.show(result_dir + '/' + "fluid.jpg")
            gen_arrs()
            gui.arrow_field(arrs.to_numpy())
            gui.show(result_dir + '/' + "velocity.jpg")
        elif gui.event.key == 'r':
            if (is_recording):
                print("Stop Recording")
                video_manager.make_video(gif=False, mp4=True)
            else:
                print("Start recording")
            is_recording = not is_recording
    i, j = int(x * N), int(y * N)
    if mouse_state == 1:
        v_prev[i, j] = force * ti.Vector([x - omx, omy - y])
        omx, omy = x, y 
    elif mouse_state == 2:
        dens_prev[i, j] = source
        omx, omy = x, y
    vel_step()
    dens_step()
    if (dvel):
        gen_arrs()
        gui.arrow_field(arrs.to_numpy())
    else:
        copy2pixels()
        gui.set_image(pixels)
    gui.show()
    if (is_recording):
        video_manager.write_frame(pixels)