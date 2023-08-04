import taichi as ti

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
color_step = 100
palette = ti.Vector.field(3, ti.f64, shape=(color_step,))


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
def calc_pallete():
    for i in palette:
        palette[i] = hsv_to_rgb(i / palette.shape[0], 1.0, 1.0)


calc_pallete()

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
            point[i, j].color = palette[point[i, j].escape % color_step]


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
while window.running:
    zoom = gui.slider_float("zoom", zoom, 1.0, 100000.0)
    # center_x = gui.slider_float("center.x", center_x, -2.0, 2.0)
    # center_y = gui.slider_float("center.y", center_y, -2.0, 2.0)
    max_iter = gui.slider_int("max_iter", max_iter, 1, 2000)
    if gui.button("reset"):
        zoom = 1.0
        center_x = 0.0
        center_y = 0.0
        draw = True

    if draw:
        gen_image(center_x, center_y, zoom, max_iter)
        down_sample()
        draw = False
    canvas.set_image(pixel)

    mouse = window.get_cursor_pos()
    x = left[None] + mouse[0] * 4 / zoom
    y = bottom[None] + mouse[1] * 4 / zoom
    gui.text(f"{x},{y}")

    if window.is_pressed(ti.ui.LMB):
        if window.is_pressed(ti.ui.CTRL) and action == "":
            action = "zoom-in"
        if window.is_pressed(ti.ui.ALT) and action == "":
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
            action = ""
            draw = True
        if action == "zoom-out":
            zoom /= 1.5
            center_x, center_y = x, y
            action = ""
            draw = True
        dragging = False

    window.show()
