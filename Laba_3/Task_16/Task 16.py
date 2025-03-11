from PIL import Image, ImageOps
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


def draw_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, image, color, z_buf):
    p_x0 = 500 + 20000 * x0/(z0+2)
    p_y0 = 500 + 20000 * y0/(z0+2)
    p_x1 = 500 + 20000 * x1/(z1+2)
    p_y1 = 500 + 20000 * y1/(z1+2)
    p_x2 = 500 + 20000 * x2/(z2+2)
    p_y2 = 500 + 20000 * y2/(z2+2)
    pixel = image.load()
    x_min = max(0, int(min(p_x0, p_x1, p_x2)))
    x_max = min(image.width-1 , int(max(p_x0, p_x1, p_x2)))
    y_min = max(0, int(min(p_y0, p_y1, p_y2)))
    y_max = min(image.height -1, int(max(p_y0, p_y1, p_y2)))
    for x in range(x_min, x_max+1):
        for y in range(y_min, y_max+1):
            l_0, l_1, l_2 = barycentric_coordinates(p_x0, p_y0, p_x1, p_y1, p_x2, p_y2, x, y)
            if l_0>=0 and l_1 >= 0 and l_2 >= 0:
                z_strich = l_0 *z0 + l_1*z1 + l_2*z2
                if(z_strich < z_buf[x, y]):
                    pixel[x, y] = color
                    z_buf[x, y] = z_strich

    return image

def no_face_gran(normal):
    l = np.array([0, 0, 1])
    mod_l = np.linalg.norm(l)
    mod_norma = np.linalg.norm(normal)
    cos_a = np.dot(l, normal)/(mod_l * mod_norma)
    return cos_a

def rotation_for_x(alpha):
    rad = np.radians(alpha)
    return np.array([
        [np.cos(rad), np.sin(rad) , 0],
        [- np.sin(rad), np.cos(rad), 0],
        [0, 0, 1],
    ])

def rotation_for_y(betta):
    rad = np.radians(betta)
    return np.array([
        [np.cos(rad), 0 , np.sin(rad)],
        [0, 1, 0],
        [-np.sin(rad), 0, np.cos(rad)],
    ])

def rotation_for_z(gamma):
    rad = np.radians(gamma)
    return np.array([
        [1, 0 , 0],
        [0, np.cos(rad), np.sin(rad)],
        [0, -np.sin(rad), np.cos(rad)],
    ])

def main():
    file_path = r'model_1.obj'
    arr_vertex, arr_f = parsic(file_path)
    rot_x = rotation_for_x(30)
    rot_y = rotation_for_y(45)
    rot_z = rotation_for_z(90)
    rotation_matrix = rot_x @ rot_y @ rot_z
    for i in range (len(arr_vertex)):
        arr_vertex[i] = np.dot(rotation_matrix, arr_vertex[i])
    image_size = (1000, 1000)
    image = Image.new("RGB", image_size, "white") # Создаем черное изображение (0 - черный, 255 - белый)
    z_buf = np.full(image_size, np.inf)
    # Получим точки для соединения
    for face in arr_f:
        x0 = arr_vertex[face[0] - 1][0]
        x1 = arr_vertex[face[1] - 1][0]
        x2 = arr_vertex[face[2] - 1][0]
        y0 = arr_vertex[face[0] - 1][1]
        y1 = arr_vertex[face[1] - 1][1]
        y2 = arr_vertex[face[2] - 1][1]
        z0 = arr_vertex[face[0] - 1][2]
        z1 = arr_vertex[face[1] - 1][2]
        z2 = arr_vertex[face[2] - 1][2]
        # считаем нормаль
        norma = normal(x0, y0, x1, y1, x2, y2, z0, z1, z2)
        cos_deg = no_face_gran(norma)
        if cos_deg <0:
            color = (0, 0, int(-255 * cos_deg))
            draw_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, image, color, z_buf)
        #print("Норма вектора: ", norma)
        # Рисуем треугольник
    image = ImageOps.flip(image)
    image.save('Chmonya_5.png')
    image.show()
    return image

if __name__ == "__main__":
    main()