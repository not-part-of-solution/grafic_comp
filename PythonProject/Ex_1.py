import numpy as np
from PIL import Image
#Создать матрицу размера H*W, заполнить её элементы нулевыми
#значениями, сохранить в виде полутонового (одноканального) 8-битового
#изображения высотой H и шириной W, убедиться, что полученное
#изображение открывается средствами операционной системы и полностью
#чёрное.
def main():
    H = int(input("Введите значение высоты: "))
    W = int(input("Введите значение ширины: "))
    matrix = np.zeros((H, W), dtype=np.uint8)
    image = Image.fromarray(matrix)
    image.save("black_image.png", mode = 'L') #последняя часть - затычка
    image.show()

if __name__ =="__main__":
    main()

