from PIL import Image, ImageOps
import random
import numpy as np

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
                    arr_f.append(([int(part.split('/')[0]) for part in parts[1:]])) #int

    return arr_vertex, arr_f

def barycentric_coordinates(x0, y0, x1, y1, x2, y2, x, y):

    lambda_0 = ((x - x2)* (y1-y2) - (x1 - x2)*(y - y2))/((x0 - x2)*(y1-y2) - (x1 - x2)*(y0 - y2))
    lambda_1 = ((x0 - x2)*(y - y2) - (x - x2)*(y0 - y2))/((x0 - x2)*(y1-y2) - (x1 - x2)*(y0 - y2))
    lambda_2 = 1.0 - lambda_0 - lambda_1
    return lambda_0, lambda_1, lambda_2

def normal(x0, y0, x1, y1, x2, y2, z0, z1, z2):
    arr_normal = []
    A = np.array([x1 - x2, y1 - y2, z1 - z2 ])
    #заводим вектор A(X1 - X2)
    B = np.array([x1 - x0, y1 - y0, z1 - z0 ])
    # заводим вектор A(X1 - X0)
    norma = np.cross(A, B)
    return norma


def draw_triangle(x0, y0, x1, y1, x2, y2, image, color):
    #color = (random.randint(0,255),random.randint(0, 255), random.randint(0,255))
    #color = (0, 255, 0)
    pixel = image.load()
    x_min = max(0, int(min(x0, x1, x2)))
    x_max = min(image.width-1 , int(max(x0, x1, x2)))
    y_min = max(0, int(min(y0, y1, y2)))
    y_max = min(image.height -1, int(max(y0, y1, y2)))
    for x in range(x_min, x_max+1):
        for y in range(y_min, y_max+1):
            l_0, l_1, l_2 = barycentric_coordinates(x0, y0, x1, y1, x2, y2, x, y)
            if l_0>=0 and l_1 >= 0 and l_2 >= 0:
                pixel[x, y] = color

    return image

def no_face_gran(normal):
    l = np.array([0, 0, 1])
    mod_l = np.linalg.norm(l)
    mod_norma = np.linalg.norm(normal)
    cos_a = np.dot(l, normal)/(mod_l * mod_norma)
    return cos_a

def main():
    file_path = r'model_1.obj'
    arr_vertex, arr_f = parsic(file_path)

    image_size = (1000, 1000)
    image = Image.new("RGB", image_size, "white") # Создаем черное изображение (0 - черный, 255 - белый)

    # Получим точки для соединения
    for face in arr_f:
        x0 = 500 + 5000 * arr_vertex[face[0] - 1][0]
        x1 = 500 + 5000 * arr_vertex[face[1] - 1][0]
        x2 = 500 + 5000 * arr_vertex[face[2] - 1][0]
        y0 = 500 + 5000 * arr_vertex[face[0] - 1][1]
        y1 = 500 + 5000 * arr_vertex[face[1] - 1][1]
        y2 = 500 + 5000 * arr_vertex[face[2] - 1][1]
        z0 = 500 + 5000 * arr_vertex[face[0] - 1][2]
        z1 = 500 + 5000 * arr_vertex[face[1] - 1][2]
        z2 = 500 + 5000 * arr_vertex[face[2] - 1][2]
        # считаем нормаль
        norma = normal(x0, y0, x1, y1, x2, y2, z0, z1, z2)
        cos_deg = no_face_gran(norma)
        if cos_deg <0:
            color = (int(-255 * cos_deg), 0, 0)
            draw_triangle(x0, y0, x1, y1, x2, y2, image, color)
        #print("Норма вектора: ", norma)
        # Рисуем треугольник
    image = ImageOps.flip(image)
    image.save('Rabbit_grad.png')
    image.show()
    return image

if __name__ == "__main__":
    main()