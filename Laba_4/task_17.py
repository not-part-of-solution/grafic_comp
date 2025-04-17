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
                    arr_f.append([int(part.split('/')[0]) for part in parts[1:]])
    return arr_vertex, arr_f


def barycentric_coordinates(x0, y0, x1, y1, x2, y2, x, y):
    lambda_0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2) + 1e-10)
    lambda_1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2) + 1e-10)
    lambda_2 = 1.0 - lambda_0 - lambda_1
    return lambda_0, lambda_1, lambda_2


def calc_normals(arr_vertex, arr_f):
    vertex_normals = [np.zeros(3) for _ in range(len(arr_vertex))]

    for face in arr_f:
        v0_idx, v1_idx, v2_idx = face[0] - 1, face[1] - 1, face[2] - 1
        v0, v1, v2 = arr_vertex[v0_idx], arr_vertex[v1_idx], arr_vertex[v2_idx]

        edge1 = np.array(v1) - np.array(v0)
        edge2 = np.array(v2) - np.array(v0)
        face_normal = np.cross(edge1, edge2)

        # Нормализуем нормаль к грани
        norm = np.linalg.norm(face_normal)
        if norm > 0:
            face_normal = face_normal / norm

        # Добавляем нормаль к каждой вершине
        vertex_normals[v0_idx] += face_normal
        vertex_normals[v1_idx] += face_normal
        vertex_normals[v2_idx] += face_normal

    # Нормализуем нормали вершин
    for i in range(len(vertex_normals)):
        norm = np.linalg.norm(vertex_normals[i])
        if norm > 0:
            vertex_normals[i] = vertex_normals[i] / norm

    return vertex_normals


def calculate_vertex_intensities(vertex_normals, face):
    light_dir = np.array([0, 0, 1])
    light_dir_n = np.linalg.norm(light_dir)

    v0_idx, v1_idx, v2_idx = face[0] - 1, face[1] - 1, face[2] - 1
    n0 = vertex_normals[v0_idx]
    n1 = vertex_normals[v1_idx]
    n2 = vertex_normals[v2_idx]

    i0 = np.dot(n0, light_dir)/(light_dir_n*np.linalg.norm(n0))
    i1 = np.dot(n1, light_dir)/(light_dir_n*np.linalg.norm(n1))
    i2 = np.dot(n2, light_dir)/(light_dir_n*np.linalg.norm(n2))

    return i0, i1, i2


def draw_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, image, z_buf, vertex_normals, face):
    inter = 3500
    p_x0 = 500 + inter * x0 / (z0 + 1)
    p_y0 = 500 + inter * y0 / (z0 + 1)
    p_x1 = 500 + inter * x1 / (z1 + 1)
    p_y1 = 500 + inter * y1 / (z1 + 1)
    p_x2 = 500 + inter * x2 / (z2 + 1)
    p_y2 = 500 + inter * y2 / (z2 + 1)

    pixel = image.load()
    x_min = max(0, int(min(p_x0, p_x1, p_x2)))
    x_max = min(image.width - 1, int(max(p_x0, p_x1, p_x2)))
    y_min = max(0, int(min(p_y0, p_y1, p_y2)))
    y_max = min(image.height - 1, int(max(p_y0, p_y1, p_y2)))

    i0, i1, i2 = calculate_vertex_intensities(vertex_normals, face)

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            l0, l1, l2 = barycentric_coordinates(p_x0, p_y0, p_x1, p_y1, p_x2, p_y2, x, y)
            if l0 >= 0 and l1 >= 0 and l2 >= 0:
                z_strich = l0 * z0 + l1 * z1 + l2 * z2
                if z_strich < 0:  # Отсечение задних граней
                    continue
                if z_strich < z_buf[x, y]:
                    intensity = l0 * i0 + l1 * i1 + l2 * i2
                    intensity = max(0, min(1, intensity))  # Ограничение [0,1]
                    color_val = int(255 * intensity)
                    pixel[x, y] = (color_val, color_val, color_val)  # Серый цвет для лучшего восприятия
                    z_buf[x, y] = z_strich


def rotation_for_x(alpha):
    rad = np.radians(alpha)
    return np.array([
        [1, 0, 0],
        [0, np.cos(rad), -np.sin(rad)],
        [0, np.sin(rad), np.cos(rad)]
    ])


def rotation_for_y(betta):
    rad = np.radians(betta)
    return np.array([
        [np.cos(rad), 0, np.sin(rad)],
        [0, 1, 0],
        [-np.sin(rad), 0, np.cos(rad)]
    ])

def rotation_for_z(gamma):
    rad = np.radians(gamma)
    return np.array([
        [np.cos(rad), -np.sin(rad), 0],
        [np.sin(rad), np.cos(rad), 0],
        [0, 0, 1],
    ])


def main():
    file_path = 'model_1.obj'
    arr_vertex, arr_f = parsic(file_path)

    # Поворот модели
    rot_x = rotation_for_x(30)
    rot_y = rotation_for_y(15)
    rot_z = rotation_for_z(0)
    rotation_matrix = rot_z @ rot_y @ rot_x

    rotated_vertices = []
    for vertex in arr_vertex:
        rotated_vertex = np.dot(rotation_matrix, vertex)
        rotated_vertices.append(rotated_vertex)
    arr_vertex = rotated_vertices

    image_size = (1000, 1000)
    vertex_normals = calc_normals(arr_vertex, arr_f)
    image = Image.new("RGB", image_size, "blue")  # Черный фон для лучшего контраста
    z_buf = np.full((image_size[1], image_size[0]), np.inf)

    for face in arr_f:
        if len(face) < 3:
            continue

        x0, y0, z0 = arr_vertex[face[0] - 1]
        x1, y1, z1 = arr_vertex[face[1] - 1]
        x2, y2, z2 = arr_vertex[face[2] - 1]

        edge1 = np.array([x1 - x0, y1 - y0, z1 - z0])
        edge2 = np.array([x2 - x0, y2 - y0, z2 - z0])
        normal = np.cross(edge1, edge2)
        cos_deg = no_face_gran(normal)
        if cos_deg > 0:
            draw_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, image, z_buf, vertex_normals, face)

    image = ImageOps.flip(image)
    image.save('Chmonya_5_gouraud.png')
    image.show()
    return image

def no_face_gran(normal):
    l = np.array([0, 0, 1])
    mod_l = np.linalg.norm(l)
    mod_norma = np.linalg.norm(normal)
    cos_a = np.dot(l, normal)/(mod_l * mod_norma)
    return cos_a


if __name__ == "__main__":
    main()
