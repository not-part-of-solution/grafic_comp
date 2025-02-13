import numpy as np
from PIL import Image
#Создать матрицу размера H*W, заполнить её элементы значениями, равными
#255, сохранить в виде полутонового (одноканального) 8-битового
#изображения высотой H и шириной W, убедиться, что полученное
#изображение открывается средствами операционной системы и полностью
#белое.
def main():
    H = int(input("Введите значение высоты: "))
    W = int(input("Введите значение ширины: "))
    matrix = np.ones((H, W), dtype=np.uint8) * 255
    image = Image.fromarray(matrix)
    image.save('white_image.png')
    image.show()

if __name__ =="__main__":
    main()

