from PIL import Image, ImageOps
import numpy as np
def bresenham_line(image, x0, y0, x1, y1):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = 2 * abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if xchange:
            image[x, y] = 0
        else:
            image[y, x] = 0
        derror += dy
        if (derror > (x1 - x0)):
            derror -= 2 * (x1 - x0)
            y += y_update


def draw_vertex(arr, image_size = (1000,1000), mult = 2000, up_down= 500):
    image = Image.new("RGB", image_size, "white")
    pixel = image.load()
    color = (255, 0, 0)
    for vertex in arr:
        x, y, z = vertex
        x_pixel = int(mult * x + up_down)
        y_pixel = int(mult * y + up_down)
        pixel[x_pixel, y_pixel] = color

    return image

def parsic(file_path):

    arr_vertex = []
    arr_f = []
    with open(file_path, 'r') as objFile:
        for line in objFile:
            parts = line.strip().split()
            if not parts:  # Пропускаем пустые строки
                continue
            type = parts[0]
            if type == "v":
                x, y, z = map(float, parts[1:4])
                arr_vertex.append((x, y, z))
            elif type == "f":
                if len(parts) > 1:
                    part_1 = ' '.join(parts[1:2])  #подробили на составные части
                    part_2 = ' '.join(parts[2:3])
                    part_3 = ' '.join(parts[3:4])
                    arr_f.append(([int(part.split('/')[0]) for part in parts[1:]])) #int

    return arr_vertex, arr_f

def main():
    file_path = r'model_1.obj'
    arr_vertex, arr_f = parsic(file_path)

    pixel = np.ones((1000, 1000), dtype=np.uint8)  # Создаем черное изображение (0 - черный, 255 - белый)

    # Получим точки для соединения
    for i in range(len(arr_f)):
        x0 = 500 + 5000 * arr_vertex[arr_f[i][0] - 1][0]
        x1 = 500 + 5000 * arr_vertex[arr_f[i][1] - 1][0]
        x2 = 500 + 5000 * arr_vertex[arr_f[i][2] - 1][0]
        y0 = 500 + 5000 * arr_vertex[arr_f[i][0] - 1][1]
        y1 = 500 + 5000 * arr_vertex[arr_f[i][1] - 1][1]
        y2 = 500 + 5000 * arr_vertex[arr_f[i][2] - 1][1]
        bresenham_line(pixel, int(x0), int(y0), int(x1), int(y1))
        bresenham_line(pixel, int(x2), int(y2), int(x0), int(y0))
        bresenham_line(pixel, int(x1), int(y1), int(x2), int(y2))

    img = Image.fromarray(pixel * 255)  # Преобразование массива в изображение (умножаем на 255 для белого)
    img = ImageOps.flip(img)
    img.save('Caraculi_2.png')
    img.show()
    return img

if __name__ == "__main__":
    main()
