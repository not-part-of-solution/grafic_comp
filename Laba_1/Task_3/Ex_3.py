def parsic(file_path):

    arr = []
    with open(file_path, 'r') as objFile:
        for line in objFile:
            parts = line.strip().split()
            if not parts:  #Пропускаем пустые строки
                continue
            type = parts[0]
            if type == "v":
                x, y, z = map(float, parts[1:4])
                arr.append((x, y, z))
    return arr

def main():
    file_path = r'model_1.obj'
    arr = parsic(file_path)
    for i in range(0, len(arr)):
        print("Точка", i+1, "имеет координаты: ", arr[i])

if __name__ == "__main__":
    main()

