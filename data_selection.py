# -*- coding: utf-8 -*-
"""
Created by Dufy on 2019/11/27  11:21
IDE used: PyCharm 
Description :
1)对总的数据集进行划分
2)输出：'test_split_data.txt'
       ‘train_split_data.txt'
Remark:   记得在运行前，清空  'test_split_data.txt' 与 ‘train_split_data.txt'内容
"""
import os
import random
import linecache
import matplotlib.pyplot as plt
# from Classifier.DataPretreatment import load_label_name_map,load_stop_word_list

# random.seed(42)  # 设置随机数，用于保证每次随机生成得到的数据是一致的
# stop_words = load_stop_word_list("stopwords.txt")
# print(stop_words)
# f_train = open('train_split_data.txt', 'w')
# f_train.truncate()
# f_train.close()
# f_test = open('test_split_data.txt', 'w')
# f_test.truncate()
# f_test.close()  # 记得在运行前，清空  'test_split_data.txt' 与 ‘train_split_data.txt'内容
# f_all = open('all_labels.txt', 'w')
# f_all.truncate()
# f_all.close()

class Operate_txt:
    def __init__(self, url):
        self.url = url
    def txt_write_line(self, save_path, line): #将某一行写入txt
        filenames = save_path
        # print("wenjianm:" + filenames)
        # fs_list = []
        try:
            # fs_list.append(open(filenames, 'a', encoding='utf-8'))  # a 指定打开文件的模式，a为追加   r为只读
            with open(filenames, mode='a+', encoding='utf-8') as f:
                # f.write(line + '\n')
                f.write(line)
        except IOError as ex:
            print(ex)
            print('\033[1;31m 写文件时发生错误!\033[0m')

    def txt_get_somelines(self, lines_number):
        print(lines_number)
        file_names = tuple(os.listdir(self.label_path))  # 转为tuple
        fs_list = []
        label_number = {}
        for filename in file_names:
            fs_list.append(open(filename, 'w', encoding='utf-8'))
        # for i in range(len(file_names)):  # 遍历各txt
        for i, name0 in enumerate(file_names):  # 遍历各txt
            txt_path = self.label_path + '\\' + name0
            print('读取', txt_path)
            txt = open(txt_path, 'rb')

            data = txt.read().decode('utf-8')  # python3一定要加上这句不然会编码报错！
            txt.close()
            n = data.count('\n')
            label_number[name0.replace('.txt','')] = n
            # n += 1
            print("总行数:", n)
            num = list(range(1, n))
            # test_size = round(size * n)
            name_ = name0.replace('.txt', '')
            label_name0 = '__label__' + name_ + ' , '
            if lines_number < n:
                test_slice = random.sample(num, lines_number)  # 从list中随机获取个元素，作为一个片断返回
                # print('测试取值：{}'.format(test_slice))
                for i in num:
                    # print(linecache.getline(txt_path, i))
                    # if linecache.getline(txt_path, i):
                    line = label_name0 + linecache.getline(txt_path, i)  # 待写入文件行
                    # print(line)
                    # if linecache.getline(txt_path, i):
                    if i in test_slice:  # 如果在随机取的值里
                        # print('test')
                        self.txt_write_line('selection_data.txt', line)
            else:  # 直接全部写进去
                num.append(n)
                for i in num:
                    line = label_name0 + linecache.getline(txt_path, i)  # 待写入文件行
                    self.txt_write_line('selection_data.txt', line)
        print(label_number)
        for i in label_number:
            #     print(i, a[i])
            plt.plot(i, label_number[i], 'r:o')
        plt.grid()
        plt.show()
        print('txt_get_somelines: done!!!!')

    def txt_change(self, merge_path):  # 对每一个txt进行原地改变
        self.label_path = merge_path
        labels_name = []
        lines_nunber = []
        # merge_path = r'D:\dufy\code\2019-11-25\data\class-labels'
        file_names = tuple(os.listdir(merge_path))  # 转为tuple
        fs_list = []
        print('file_names：{}，类型{}'.format(file_names, type(file_names)))
        for i, name0 in enumerate(file_names):
            print(i, name0, type(name0))
        try:
            for filename in file_names:
                fs_list.append(open(filename, 'w', encoding='utf-8'))
            # for i in range(len(file_names)):  # 遍历各txt
            for i, name0 in enumerate(file_names): # 遍历各txt
                print(i, name0, type(name0))
                txt_path = merge_path + '\\' + name0
                print(txt_path)
                j = 0
                # with open(txt_path, mode='r', encoding='utf-8') as f:
                #     for line in f:
                #         print(line, end='')  # end=' '意思是末尾不换行，加空格。
                #         self.txt_write_line('all_labels.txt', line)
                #         # time.sleep(0.1)
                #         j += 1
                #     self.txt_write_line('all_labels.txt', '\n')
                with open(txt_path, 'r', encoding='utf-8') as r:
                    print(txt_path)
                    lines = r.readlines()
                with open(txt_path, 'w+', encoding='utf-8') as w:
                    for aa in lines:
                        w.write(aa)

        except IOError as ex:
            print(ex)
            print('写文件时发生错误!')
        finally:
            for fs in fs_list:
                fs.close()
        print('操作完成!')

    # def txt_print(self):
    #     i = 0
    #     with open(self.url, mode='r', encoding='utf-8') as f:
    #         for line in f:
    #             print(line, end='')  # end=' '意思是末尾不换行，加空格。
    #             # time.sleep(0.1)
    #             i += 1
    #     print()
    #     print('\033[1;32m txt行数：{}\033[0m'.format(i))
    #     with open(self.url, 'r', encoding='utf-8') as r:
    #         lines = r.readlines()
    #     with open(self.url, 'w+', encoding='utf-8') as w:
    #         for aa in lines:
    #             aa = aa.lower()
    #             for word in aa:
    #                 # print(word)
    #                 if str(word) in stop_words:
    #                     aa = aa.replace(word, '')
    #             # aa.strip()
    #             # for i in aa:
    #             #     print(i)
    #             w.write(aa)


def merge_txts(line_select_number):
    f_all1 = open('selection_data.txt', 'w')
    f_all1.truncate()
    f_all1.close()
    a = Operate_txt(r'D:\dufy\code\2019-11-29\data\方案验证板.txt')  # 添加需要操作的文件路径
    # a.txt_print()
    # a.txt_write(r'data\AA.txt')   # 写入文件路径

    # a.txt_split(0.5)  # 测试比例
    # a.txt_change(r'D:\dufy\code\ft_BOM\data\initial')  # 待融合txt 合剂路径
    a.txt_change(r'D:\dufy\code\work_record\data\excel_write')  # 待融合txt 合剂路径
    a.txt_get_somelines(line_select_number)  # 读取行数

if __name__ == '__main__':

    a = Operate_txt(r'D:\dufy\code\2019-11-29\data\方案验证板.txt')  # 添加需要操作的文件路径
    # a.txt_print()
    # a.txt_write(r'data\AA.txt')   # 写入文件路径

    # a.txt_split(0.5)  # 测试比例
    # a.txt_change(r'D:\dufy\code\ft_BOM\data\initial')  # 待融合txt 合剂路径
    a.txt_change(r'D:\dufy\code\ft_BOM\data\select_example')  # 待融合txt 合剂路径
    # a.txt_get_somelines(1500)   # 读取行数
    a.txt_get_somelines(500)   # 读取行数
# 'selection_data.txt',  D:\dufy\code\ft_BOM
