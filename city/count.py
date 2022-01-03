import os


def main():
    data_root = '/data1/2019/HaoL/ldh/irn/city/images'

    '''
    {'road': 2520, 'building': 2399, 'vegetation': 1915, 'sky': 882}
    {'road building vegetation sky ': 727, 'road building vegetation ': 815, 'road building ': 627, 'road building sky ': 44, 'road ': 10, 'road vegetation ': 194, 'road vegetation sky ': 96, 'building ': 114, 'building vegetation ': 54, 'building vegetation sky ': 11, 'vegetation ': 11}
    
    {'road': 4407, 'building': 3606, 'vegetation': 2344, 'sky': 1119}
    {'road building vegetation sky ': 887, 'road vegetation ': 353, 'road ': 1580, 'road building ': 1071, 'road building sky ': 68, 'road building vegetation ': 349, 'building ': 716, 'vegetation ': 147, 'building vegetation ': 450, 'road vegetation sky ': 92, 'building vegetation sky ': 49, 'building sky ': 8, 'vegetation sky ': 9}

    
    '''
    count_dic = {'road': 0, 'building': 0, 'vegetation': 0, 'sky': 0}       # ???????
    count_dic_multi_class = {}                                              # ???????

    id_to_class = {0: 'road', 1: 'building', 2: 'vegetation', 3: 'sky'}
    for root, dirs, files in os.walk(data_root):
        # print(root, dirs, files)
        for file in files:
            _, label_str = file.split('_class_')                # label_str:'012.png'
            labels = [int(idx) for idx in label_str[:-4]]       # [0,1,2]
            multi_class_name = ''
            for label in labels:
                count_dic[id_to_class[label]] += 1             # ?????????
                multi_class_name += id_to_class[label]+" "    # ??????

            if multi_class_name not in count_dic_multi_class:
                count_dic_multi_class[multi_class_name] = 1
            else:
                count_dic_multi_class[multi_class_name] += 1
    print(count_dic)
    print(count_dic_multi_class)


if __name__ == '__main__':
    main()
