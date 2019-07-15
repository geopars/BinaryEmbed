def main():
    file_path = '/Users/Krish/Downloads/query_asin_test.data'

    query_list = []
    with open(file_path) as f:
        for line in f:
            items = line.split(',')
            # print(items)
            query_list.append(items[0])


    query_set = list(set(query_list))
    print(query_set)

    f = open("topK.data", "w+")

    for i in range(len(query_set)):
        f.write(query_set[i] + "\n")
        list1 = TopMatch(query_set[i], file_path)
        f.write(str(list1) + "\n")


def TopMatch(query, file_path):
    map = {}
    top_list = []

    with open(file_path) as tf:
        for line in tf:
            items = line.split(',')

            if query == items[0]:
                if items[1] in map:
                    x = map.get(items[1])
                    map[items[1]] = x + 1
                else:
                    map[items[1]] = 1

    list = [v[0] for v in sorted(map.items(), key=lambda kv: (-kv[1], kv[0]))]

    for i in range(min(100, len(list))):
        top_list.append((list[i], map.get(list[i])))
    # print(len(list))
    # ret = list[:min(100, len(list))]
    return top_list


if __name__ == '__main__':
    main()