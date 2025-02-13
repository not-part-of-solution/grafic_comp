from PIL import Image, ImageOps

def pars_for_me(file_path):

    arr_vertex = []
    with open(file_path, 'r') as objFile:
        for line in objFile:
            parts = line.strip().split()
            if not parts:  # Пропускаем пустые строки
                continue
            type = parts[0]
            if type == "v":
                x, y, z = map(float, parts[1:4])
                arr_vertex.append((x, y, z))

    return arr_vertex

def draw_vertex(arr, image_size = (1000,1000), mult = 2000, up_down= 500): #рисуем вершинки
    image = Image.new("RGB", image_size, "white")
    pixel = image.load()
    color = (255, 0, 0)
    for vertex in arr:
        x, y, z = vertex
        x_pixel = int(mult * x + up_down)  #масштабируем
        y_pixel = int(mult * y + up_down)
        pixel[x_pixel, y_pixel] = color

    return image


def main():
    file_path = r'model_1.obj'
    arr = pars_for_me(file_path)
    image = draw_vertex(arr)
    image = ImageOps.flip(image) #повернули зайца
    image.save('Caraculi.png')
    image.show()

if __name__ == "__main__":
    main()

