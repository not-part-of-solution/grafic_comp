def pars_for_me(file_path):

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

def main():
    file_path = r'model_1.obj'
    arr_f = pars_for_me(file_path)
    for i in range(0, len(arr_f)):
        print(arr_f[i])

if __name__ == "__main__":
    main()
