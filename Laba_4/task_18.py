from PIL import Image, ImageOps
import numpy as np


def parsic(file_path):
    arr_vertex = []
    arr_f = []
    arr_vt = []
    arr_vn = []

    with open(file_path, 'r') as objFile:
        for line in objFile:
            parts = line.strip().split()
            if not parts:
                continue
            type = parts[0]

            if type == "v":
                x, y, z = map(float, parts[1:4])
                arr_vertex.append((x, y, z))
            elif type == "vt":
                u, v = map(float, parts[1:3])
                arr_vt.append((u, v))
            elif type == "vn":
                x, y, z = map(float, parts[1:4])
                arr_vn.append((x, y, z))
            elif type == "f":
                if len(parts) > 1:
                    face = []
                    for part in parts[1:]:
                        indices = part.split('/')
                        # Обработка случаев: f v, f v/vt, f v//vn, f v/vt/vn
                        vertex_idx = int(indices[0])
                        texcoord_idx = int(indices[1])
                        normcoord_idx = int(indices[2])
                        face.append((vertex_idx, texcoord_idx, normcoord_idx))
                    arr_f.append(face)

    return arr_vertex, arr_f, arr_vt, arr_vn


def barycentric_coordinates(x0, y0, x1, y1, x2, y2, x, y):
    lambda_0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda_1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda_2 = 1.0 - lambda_0 - lambda_1
    return lambda_0, lambda_1, lambda_2


def normal(x0, y0, x1, y1, x2, y2, z0, z1, z2):
    A = np.array([x1 - x2, y1 - y2, z1 - z2])
    B = np.array([x1 - x0, y1 - y0, z1 - z0])
    return np.cross(A, B)


def calculate_vertex_normals(arr_vertex, arr_f):

    vertex_normals = [np.zeros(3) for _ in range(len(arr_vertex))]
    face_count = [0] * len(arr_vertex)

    for face in arr_f:
   
        v0_idx = face[0][0] - 1
        v1_idx = face[1][0] - 1
        v2_idx = face[2][0] - 1

        x0, y0, z0 = arr_vertex[v0_idx]
        x1, y1, z1 = arr_vertex[v1_idx]
        x2, y2, z2 = arr_vertex[v2_idx]


        face_normal = normal(x0, y0, x1, y1, x2, y2, z0, z1, z2)

  
        vertex_normals[v0_idx] += face_normal
        vertex_normals[v1_idx] += face_normal
        vertex_normals[v2_idx] += face_normal

   
        face_count[v0_idx] += 1
        face_count[v1_idx] += 1
        face_count[v2_idx] += 1


    for i in range(len(vertex_normals)):
        if face_count[i] > 0:
            vertex_normals[i] = vertex_normals[i] / np.linalg.norm(vertex_normals[i])

    return vertex_normals


def calculate_light_intensity(normal, light_direction=np.array([0, 0, 1])):

    normal = normal / np.linalg.norm(normal)
    light_direction = light_direction / np.linalg.norm(light_direction)


    intensity = np.dot(normal, light_direction)


    intensity = max(0, min(1, intensity))
    ambient = 0.2
    intensity = ambient + (1 - ambient) * intensity

    return intensity


def draw_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2,
                  u0, v0, u1, v1, u2, v2,
                  n0, n1, n2,  
                  image, z_buf, texture_img):
    inter = 3500
    scale = image.width / 2
    p_x0 = scale + inter * x0 / (z0 + 1)
    p_y0 = scale + inter * y0 / (z0 + 1)
    p_x1 = scale + inter * x1 / (z1 + 1)
    p_y1 = scale + inter * y1 / (z1 + 1)
    p_x2 = scale + inter * x2 / (z2 + 1)
    p_y2 = scale + inter * y2 / (z2 + 1)

    pixel = image.load()
    texture = texture_img.load()
    WT, HT = texture_img.size

    x_min = max(0, int(min(p_x0, p_x1, p_x2)))
    x_max = min(image.width - 1, int(max(p_x0, p_x1, p_x2)))
    y_min = max(0, int(min(p_y0, p_y1, p_y2)))
    y_max = min(image.height - 1, int(max(p_y0, p_y1, p_y2)))


    intensity0 = calculate_light_intensity(n0)
    intensity1 = calculate_light_intensity(n1)
    intensity2 = calculate_light_intensity(n2)

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            l0, l1, l2 = barycentric_coordinates(p_x0, p_y0, p_x1, p_y1, p_x2, p_y2, x, y)
            if l0 >= 0 and l1 >= 0 and l2 >= 0:
                z = l0 * z0 + l1 * z1 + l2 * z2
                if z < z_buf[x, y]:
            
                    tex_u = l0 * u0 + l1 * u1 + l2 * u2
                    tex_v = l0 * v0 + l1 * v1 + l2 * v2

          
                    tex_x = min(WT - 1, int(tex_u * (WT - 1)))
                    tex_y = min(HT - 1, int((1 - tex_v) * (HT - 1)))
                    color = texture[tex_x, tex_y]

    
                    intensity = l0 * intensity0 + l1 * intensity1 + l2 * intensity2

    
                    shaded_color = (
                        int(min(255, color[0] * intensity * 5)), 
                        int(min(255, color[1] * intensity * 5)),
                        int(min(255, color[2] * intensity * 5))
                    )

                    pixel[x, y] = shaded_color
                    z_buf[x, y] = z


def no_face_gran(normal):
    l = np.array([0, 0, 1])
    return np.dot(l, normal) / (np.linalg.norm(l) * np.linalg.norm(normal))


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
    texture_path = 'Винстон_274.png'

    arr_vertex, arr_f, arr_vt, arr_vn = parsic(file_path)
    texture_img = Image.open(texture_path)

    # Calculate vertex normals for Gouraud shading
    vertex_normals = calculate_vertex_normals(arr_vertex, arr_f)

    # Apply rotations
    rot_x = rotation_for_x(60)
    rot_y = rotation_for_y(150)
    rot_z = rotation_for_z(0)
    rotation_matrix = rot_z @ rot_y @ rot_x

    # Rotate vertices and normals
    rotated_vertices = []
    rotated_normals = []
    for i, vertex in enumerate(arr_vertex):
        rotated_vertex = np.dot(rotation_matrix, vertex)
        rotated_vertices.append(rotated_vertex)

        # Rotate normal (using inverse transpose for correct normal transformation)
        normal_rot_matrix = np.linalg.inv(rotation_matrix).T
        rotated_normal = np.dot(normal_rot_matrix, vertex_normals[i])
        rotated_normals.append(rotated_normal)

    arr_vertex = rotated_vertices
    vertex_normals = rotated_normals

    image_size = (1000, 1000)
    image = Image.new("RGB", image_size, "black")
    z_buf = np.full(image_size, np.inf)

    for face in arr_f:
        # Get triangle vertices
        x0 = arr_vertex[face[0][0] - 1][0]
        y0 = arr_vertex[face[0][0] - 1][1]
        z0 = arr_vertex[face[0][0] - 1][2]

        x1 = arr_vertex[face[1][0] - 1][0]
        y1 = arr_vertex[face[1][0] - 1][1]
        z1 = arr_vertex[face[1][0] - 1][2]

        x2 = arr_vertex[face[2][0] - 1][0]
        y2 = arr_vertex[face[2][0] - 1][1]
        z2 = arr_vertex[face[2][0] - 1][2]

        
        u0 = arr_vt[face[0][1] - 1][0]
        v0 = arr_vt[face[0][1] - 1][1]

        u1 = arr_vt[face[1][1] - 1][0]
        v1 = arr_vt[face[1][1] - 1][1]

        u2 = arr_vt[face[2][1] - 1][0]
        v2 = arr_vt[face[2][1] - 1][1]

        
        n0 = vertex_normals[face[0][0] - 1]
        n1 = vertex_normals[face[1][0] - 1]
        n2 = vertex_normals[face[2][0] - 1]


        norma = normal(x0, y0, x1, y1, x2, y2, z0, z1, z2)
        cos_deg = no_face_gran(norma)

        if cos_deg < 0:
            draw_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2,
                          u0, v0, u1, v1, u2, v2,
                          n0, n1, n2,  # Pass vertex normals
                          image, z_buf, texture_img)

    image = ImageOps.flip(image)
    image.save('Chmonya.png')
    image.show()


if __name__ == "__main__":
    main()
