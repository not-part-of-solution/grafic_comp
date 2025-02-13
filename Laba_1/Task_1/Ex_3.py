import numpy as np
from PIL import Image

#Создать матрицу размера H*W*3, заполнить её элементы значениями,
#равными (255, 0, 0), сохранить в виде цветного (трёхканального) 8-битового
#изображения высотой H и шириной W, убедиться, что полученное
#изображение открывается средствами операционной системы и полностью
#красное.

def main():
    H = int(input("Введите высоту изображения "))
    W = int(input("Введите ширину изображения "))
    matrix = np.ones((H, W, 3), dtype=np.uint8)
    matrix[:] = [255, 0, 0]
    image = Image.fromarray(matrix, mode = 'RGB')
    image.save("red_image.png")
    image.show()

if __name__ =="__main__":
    main()
