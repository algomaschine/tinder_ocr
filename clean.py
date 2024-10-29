


def read_sections2(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()

    lines = []
    names= []
    name = False


    stop = ["lives in" , "kilometers away", "академия","woman" ,"высшее","рост", "московский", "институт", "инстаграм", "государственный","instagram", "университет", "инст", "inst", "вшэ","university"]

    for line in data:
        line = line.lower()

        if line.startswith("----------------- size"):
            name = True
            continue

        elif name:
            n = line.strip().split(" ")
            for nn in n:
                names.append(nn)
            name = False
            continue

        elif line.startswith("-----------------------------------------------"):
            continue

        elif line.strip().isdigit() or len(line.strip())==0:
            continue

        #elif any(word in line for word in stop):
        #    continue

        else:
            lines.append(line)

    return lines, names

lines, names = read_sections2("desc.txt")
for s in lines:
    #print(".")
    print(s)
    #print("..")

for n in names:
    print(n)
