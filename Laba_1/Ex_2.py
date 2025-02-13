import numpy as np
import math
from PIL import Image


def dotted_line(image, x0, x1, y0, y1, count,color):
    step = 1.0/count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t)*x0 + t*x1)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color

def dotted_line_v2(image, x0, x1, y0, y1,color):
    count = math.sqrt((x0 - x1) ** 2 + (y0 -
                                        y1) ** 2)
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color

def x_loop_line(image, x0, y0, x1, y1, color):
    for x in np.arange(x0, x1):
        t = (x-x0)/(x1-x0)
        y = round((1.0 - t)* y0 + t * y1)
        image[y, x] = color

def x_loop_line_hotfix_v1(image, x0, y0, x1, y1, color):
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in np.arange(x0, x1):
        t = (x-x0)/(x1-x0)
        y = round((1.0 - t)* y0 + t * y1)
        image[y, x] = color

def x_loop_line_hotfix_v2(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    for x in np.arange(x0, x1 +1):
        t = (x-x0)/(x1-x0)
        y = round((1.0 - t)* y0 + t * y1)

        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color

def x_loop_line_v2(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in np.arange(x0, x1):
        t = (x-x0)/(x1-x0)
        y = round((1.0 - t)* y0 + t * y1)
        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color

def x_loop_line_v2_y_no_calc(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = abs(y1 - y0) / (x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if (derror > 0.5):
            derror -= 1.0
            y += y_update

def x_loop_line_v2_y_no_calc_v2_someone_dont_know(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = 2 * (x1 - x0) * abs(y1 - y0) / (x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if (derror > 2 * (x1 - x0) * 0.5):
            derror -= 2 * (x1 - x0) * 1
            y += y_update

def bresenham_line(image, x0, y0, x1, y1, color):
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
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if (derror > (x1 - x0)):
            derror -= 2 * (x1 - x0)
            y += y_update

def main():
    height = 200 #задаем конкретную матрицу
    weight = 200
    matrix = np.ones((height, weight, 3), dtype = np.uint8)*255 #белый фончик
    color = [255, 0, 0]
    alpha = 2 * np.pi/13
    for i in range(0, 13):
        x0 = 100
        y0 = 100
        x1 = round(100 + 95 * math.cos(alpha * i))
        y1 = round(100 + 95 * math.sin(alpha * i))
        dotted_line(matrix, x0, x1, y0, y1, count=100, color=color)
        #dotted_line_v2(matrix, x0, y0, x1, y1, color=color)
        #x_loop_line(matrix, x0, y0, x1, y1, color=color)
        #x_loop_line_hotfix_v1(matrix, x0, y0, x1, y1, color=color)
        #x_loop_line_hotfix_v2(matrix, x0, y0, x1, y1, color=color)
        #x_loop_line_v2(matrix, x0, y0, x1, y1, color=color)
        #x_loop_line_v2_y_no_calc(matrix, x0, y0, x1, y1, color=color)
        #x_loop_line_v2_y_no_calc_v2_someone_dont_know(matrix, x0, y0, x1, y1, color=color)
        #bresenham_line(matrix, x0, y0, x1, y1, color=color)
    image = Image.fromarray(matrix, 'RGB') #тут вторая часть нужна, чтобы не улетело
    image.save('dotted_line.png')
    image.show()




if __name__ == "__main__":
    main()