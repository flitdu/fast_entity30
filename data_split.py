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
# random.seed(42)  # 设置随机数，用于保证每次随机生成得到的数据是一致的


# f_all = open('all_labels.txt', 'w')
# f_all.truncate()
# f_all.close()

class Operate_txt:
    def __init__(self, url):
        self.url = url

    def txt_merge(self, merge_path):  #读取多个txt合并为1个
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
                with open(txt_path, mode='r', encoding='utf-8') as f:
                    for line in f:
                        print(line, end='')  # end=' '意思是末尾不换行，加空格。
                        self.txt_write_line('all_labels.txt', line)
                        # time.sleep(0.1)
                        j += 1
                    self.txt_write_line('all_labels.txt', '\n')
                lines_nunber.append(j)
                labels_name.append(name0)
                print()
                print('\033[1;32m {}读取结束,行数：{}\033[0m'.format(name0, i))
            print(lines_nunber, labels_name)
            print("\033[1;31m 数据集共：{}条\033[0m".format(sum(lines_nunber)))
            for i in range(len(lines_nunber)):
                print('\033[1;31m __{}__数量：{},占比{:.1f}%\033[0m'.format(labels_name[i], lines_nunber[i], 100*lines_nunber[i]/sum(lines_nunber)))
                # line = linecache.getline(self.url, i)  # 待写入文件行
                # self.txt_write_line('all_labels.txt', line)
                #
                # if is_prime(number):
                #     if number < 100:
                #         fs_list[0].write('{:0>2} 是 素数 ✅ \n'.format(number))
                #     elif number < 1000:
                #         fs_list[1].write('{:0>2} 是 素数 ✅ \n'.format(number))
                #     else:
                #         fs_list[2].write(str(number) + '\n')
                # else:
                #     if number < 100:
                #         fs_list[0].write('{:0>2} 不是 素数 ❌ \n'.format(number))
                #     elif number < 1000:
                #         fs_list[1].write('{:0>2} 不是 素数 ❌ \n'.format(number))
            #                 else:
            #                     fs_list[2].write(str(number) + '\n')

        except IOError as ex:
            print(ex)
            print('写文件时发生错误!')
        finally:
            for fs in fs_list:
                fs.close()
        print('操作完成!')

    def txt_print(self):
        i = 0
        with open(self.url, mode='r', encoding='utf-8') as f:
            for line in f:
                print(line, end='')  # end=' '意思是末尾不换行，加空格。
                # time.sleep(0.1)
                i += 1
        print()
        print('\033[1;32m txt行数：{}\033[0m'.format(i))

    def txt_write(self, save_path):  #默认覆盖写入
        filenames = save_path
        fs_list = []
        try:
            fs_list.append(open(filenames, 'w', encoding='utf-8'))
            with open(self.url, mode='r', encoding='utf-8') as f:
                for line in f:
                    # fs_list[0].write(line + '\n')
                    fs_list[0].write(line)
        except IOError as ex:
            print(ex)
            print('\033[1;31m 写文件时发生错误!\033[0m')
        finally:
            for fs in fs_list:
                fs.close()
        print('\033[1;31m 写入操作完成!\033[0m')

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


    def txt_split(self, size):
        txt = open(self.url, 'rb')
        data = txt.read().decode('utf-8')  # python3一定要加上这句不然会编码报错！
        txt.close()
        n = data.count('\n')
        n += 1
        print("总行数:", n)
        num = list(range(1, n))
        test_size = round(size * n)
        test_slice = random.sample(num, test_size)  # 从list中随机获取个元素，作为一个片断返回
        train_slice = []
        for item in num:
            if item not in test_slice:
                train_slice.append(item)
        print('测试取值：{}'.format(test_slice))
        for i in num:
            line = linecache.getline(self.url, i)  # 待写入文件行
            if i%1000 == 0:
                print('进度：{:.2f}'.format(i/n))
            # print(line)
            if i in test_slice:
                # print('test')
                self.txt_write_line('test_split_data.txt', line)
            else:
                # print('train')
                self.txt_write_line('train_split_data.txt', line)

def train_datas_split():
    f_train = open('train_split_data.txt', 'w')
    f_train.truncate()
    f_train.close()
    f_test = open('test_split_data.txt', 'w')
    f_test.truncate()
    f_test.close()  # 记得在运行前，清空  'test_split_data.txt' 与 ‘train_split_data.txt'内容
    # a = Operate_txt(r'D:\dufy\code\2019-11-25\test.txt')  # 添加需要操作的文件路径
    # a = Operate_txt(r'D:\dufy\code\2019-11-25\fasttext.test1125.txt')  # 添加需要操作的文件路径
    a = Operate_txt(r'D:\dufy\code\work_record\selection_data.txt')  # 添加需要操作的文件路径
    a.txt_print()
    # a.txt_write('text_write_test.txt')   # 写入文件路径

    a.txt_split(0.25)  # 测试比例
    # a.txt_merge(r'D:\dufy\code\2019-11-25\data\class-labels')  # 待融合txt 合剂路径
    print('ENDDD!!!')

if __name__ == '__main__':

    # a = Operate_txt(r'D:\dufy\code\2019-11-25\test.txt')  # 添加需要操作的文件路径
    # a = Operate_txt(r'D:\dufy\code\2019-11-25\fasttext.test1125.txt')  # 添加需要操作的文件路径
    a = Operate_txt(r'D:\dufy\code\2019-11-25\labels30.txt')  # 添加需要操作的文件路径
    a.txt_print()
    # a.txt_write('text_write_test.txt')   # 写入文件路径

    a.txt_split(0.25)  # 测试比例
    # a.txt_merge(r'D:\dufy\code\2019-11-25\data\class-labels')  # 待融合txt 合剂路径
    print('ENDDD!!!')

