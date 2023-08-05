import taichi as ti
import time

ti.init(arch=ti.vulkan)
# ti.init(debug=True)
[width, height] = [1000, 1000]
# zoom 1: -2 to 2 is width
zoom = 1.0
center_x = 0.0
center_y = 0.0
oversample = 2
max_iter = 100

# create window
window = ti.ui.Window("Mandelbrot Viewer", (width, height))
canvas = window.get_canvas()
canvas.set_background_color((1.0, 1.0, 1.0))
gui = window.get_gui()

point = ti.Struct.field(
    {
        "escape": int,
        "color": ti.types.vector(3, ti.f64),
    },
    shape=(width * oversample, height * oversample),
)
pixel = ti.Vector.field(3, ti.f32, shape=(width, height))
palette_color_count = 200
palette = ti.Vector.field(3, ti.f64, shape=(palette_color_count,))
palette_color_step = 5
color_shift = 0


@ti.func
def hsv_to_rgb(h: ti.f64, s: ti.f64, v: ti.f64):
    result = ti.Vector([0.0, 0.0, 0.0])
    if s == 0.0:
        result = (v, v, v)
    else:
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        if i == 0:
            result = (v, t, p)
        if i == 1:
            result = (q, v, p)
        if i == 2:
            result = (p, v, t)
        if i == 3:
            result = (p, q, v)
        if i == 4:
            result = (t, p, v)
        if i == 5:
            result = (v, p, q)
    return result




@ti.kernel
def gen_palette(palette_color_step:ti.i32, color_shift:ti.i32):
    for i in palette:
        ii = i + color_shift
        palette[i] = hsv_to_rgb(
            (ii // palette_color_step) / (palette.shape[0] // palette_color_step),
            1.0,
            (2.0 - (ii % palette_color_step) / palette_color_step) / 2.0,
        )
gen_palette(palette_color_step, color_shift)

left = ti.field(dtype=ti.f64, shape=())
bottom = ti.field(dtype=ti.f64, shape=())
delta = ti.field(dtype=ti.f64, shape=())


@ti.kernel
def gen_image(center_x: ti.f64, center_y: ti.f64, zoom: ti.f64, max_iter: int):
    left[None] = center_x - 2.0 / zoom
    bottom[None] = center_y - 2.0 * height / width / zoom
    delta[None] = 4.0 / zoom / width / oversample
    for i, j in point:
        real0 = real = left[None] + i * delta[None]
        imag0 = imag = bottom[None] + j * delta[None]
        point[i, j].escape = 1
        while real * real + imag * imag <= 4 and point[i, j].escape < max_iter:
            realtemp = real * real - imag * imag + real0
            imag = 2 * real * imag + imag0
            real = realtemp
            point[i, j].escape = point[i, j].escape + 1
        if point[i, j].escape == max_iter:
            point[i, j].color = [0.0, 0.0, 0.0]
        else:
            point[i, j].color = palette[point[i, j].escape % palette_color_count]


@ti.kernel
def down_sample():
    for i, j in pixel:
        pixel[i, j] = 0.0
        for ii in range(i * oversample, (i + 1) * oversample):
            for jj in range(j * oversample, (j + 1) * oversample):
                pixel[i, j] += ti.cast(
                    point[ii, jj].color / oversample / oversample, ti.f32
                )


action = ""
draw = True
dragging = False
last_render_time = 0.0
while window.running:
    start = time.time()
    gui.begin("controls", 0, 0, 0.35, 0.4)
    gui.text("mouse drag: pan viewport")
    gui.text("CTRL + click: zoom in")
    gui.text("ALT + click: zoom out")
    gui.text(f"oversample: {oversample} x {oversample}, change this in code", (1.0, 0.0, 0.0))
    next_zoom = gui.slider_float("zoom", zoom, 1.0, 100000.0)
    # center_x = gui.slider_float("center.x", center_x, -2.0, 2.0)
    # center_y = gui.slider_float("center.y", center_y, -2.0, 2.0)
    next_max_iter = gui.slider_int("max_iter", max_iter, 1, 10000)
    next_palette_color_step = gui.slider_int("color_step", palette_color_step, 1, 100)
    next_color_shift = gui.slider_int("color_shift", color_shift, 1, palette_color_count)

    if gui.button("reset"):
        next_max_iter, next_palette_color_step, next_color_shift = 100, 5, 0
        center_x, center_y, next_zoom = 0.0, 0.0, 1.0
        draw = True

    gui.text("====Interesting Places====", (1.0, 0.0, 0.0))

    if gui.button("spiral"):
        next_max_iter, next_palette_color_step, next_color_shift = 3000, 5, 0
        center_x, center_y, next_zoom = -0.752207636756153, 0.037944413679301826, 16834.113
        draw = True

    if gui.button("elephant valley"):
        next_max_iter, next_palette_color_step, next_color_shift = 300, 20, 0
        center_x, center_y, next_zoom = 0.28664306825613495, -0.01270585494650385, 130
        draw = True

    if gui.button("flower"):
        next_max_iter, next_palette_color_step, next_color_shift = 450, 5, 134
        center_x, center_y, next_zoom = -1.9760422200854917, 0, 905130.562
        draw = True

    if gui.button("storm"):
        next_max_iter, next_palette_color_step, next_color_shift = 500, 5, 0
        center_x, center_y, next_zoom = 0.10681635446490144, 0.6373686353028029, 402280.250
        draw = True

    if next_max_iter != max_iter or next_zoom != zoom:
        max_iter = next_max_iter
        zoom = next_zoom
        draw = True
        action = 'config'

    if next_palette_color_step != palette_color_step or next_color_shift != color_shift:
        gen_palette(next_palette_color_step, next_color_shift)
        palette_color_step = next_palette_color_step
        color_shift = next_color_shift
        draw = True
        action = 'config'

    if draw:
        gen_image(center_x, center_y, zoom, max_iter)
        down_sample()
    canvas.set_image(pixel)
    if draw:
        last_render_time = time.time() - start
        draw = False

    mouse = window.get_cursor_pos()
    x = left[None] + mouse[0] * 4 / zoom
    y = bottom[None] + mouse[1] * 4 / zoom
    gui.text(f"center point:")
    gui.text(f"{center_x},{center_y}", (0.0, 1.0, 0.0))
    gui.text(f"mouse point:")
    gui.text(f"{x},{y}")
    if window.is_pressed(ti.ui.LMB):
        if window.is_pressed(ti.ui.CTRL):
            action = "zoom-in"
        if window.is_pressed(ti.ui.ALT):
            action = "zoom-out"
        if action == "":
            if not dragging:
                dragging = True
                center_x0, center_y0 = center_x, center_y
                mouse_x0, mouse_y0 = mouse[0], mouse[1]
            else:
                center_x = center_x0 - (mouse[0] - mouse_x0) * 4 / zoom
                center_y = center_y0 - (mouse[1] - mouse_y0) * 4 / zoom
                draw = True

    if not window.is_pressed(ti.ui.LMB):
        if action == "zoom-in":
            zoom *= 1.5
            center_x, center_y = x, y
            draw = True
        if action == "zoom-out":
            zoom /= 1.5
            center_x, center_y = x, y
            draw = True
        dragging = False
        action = ""
    gui.text(f"last render time: {last_render_time*1000.0:.3f}ms")
    gui.end()
    window.show()
