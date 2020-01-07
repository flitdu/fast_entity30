# -*- coding: utf-8 -*-

"""
-------------------------------------------------
Created by Dufy on 2019/11/25
IDE used: PyCharm Community Edition
Description :
1)批量读取excel
2) 读取数据时，将第一行当做表头，所以不会讲第一行作为训练数据！！！！！
-------------------------------------------------
Change Activity:
excel 如何读取特定列？？？
利用11的数据，选取特定列
-------------------------------------------------
"""

import pandas as pd
import os
from func import Function
from data_selection import Operate_txt
from Classifier.DataPretreatment import load_label_name_map,load_stop_word_list
from excel_row_judge import row_return_list

# f_all1 = open(r'data\excel_write\电容.txt', 'w')
# f_all1.truncate()
# f_all1.close()

# random.seed(42)  # 设置随机数，用于保证每次随机生成得到的数据是一致的
stop_words = load_stop_word_list("stopwords.txt")
# filePath = r'D:\dufy\业务数据\download_bom'
# file_names = os.listdir(filePath)
# for i, name0 in enumerate(file_names):
#     print(i, name0, type(name0))


class Operate_excel:
    def __init__(self, url):
        self.file_path = url   # 文件路径
        self.data_read = ''
        # self.errpath = errpath

    def txt_write_line(self, save_path, line): #将某一行写入txt
        filenames = save_path
        # print("wenjianm:" + filenames)
        # fs_list = []
        try:
            # fs_list.append(open(filenames, 'a', encoding='utf-8'))  # a 指定打开文件的模式，a为追加   r为只读
            with open(filenames, mode='a+', encoding='utf-8') as f:
                f.write(line + '\n')
                # f.write(line)
        except IOError as ex:
            print(ex)
            print('\033[1;31m 写文件时发生错误!!!!\033[0m')
            print(self.file_path)
            return False

    def excel_print(self):
        print('===========数据基本情况：===========')
        ss = pd.read_excel(self.file_path)  # 读取数据,设置None可以生成一个字典，字典中的key值即为sheet名字，此时不用使用DataFram，会报错
        print(ss.index)  # 获取行的索引名称
        print(ss.columns)  # 获取列的索引名称
        # print(ss)
        # 获取总和、得出行数和列数
        ss_count = ss.shape
        line = ss_count[0]
        row = ss_count[1]
        print('ss_count = {}, 数据记录条数：行={}, 列={}'.format(ss_count, line, row))

        aa = ''
        for j_line in range(line):  # j为行
            #     aa += '@' +'\n'
            for i in range(row):
                #         print(ss.loc[j_line].ix[i], end='')
                aa += str(ss.loc[j_line].iloc[i]) + '@'
            aa += '\n'
        print(type(aa))
        # for line in aa.splitlines():  # 对字符串按行读取
        #     print(line)
        self.data_read = aa   # 生成读取数据，为后续操作准备

    def excel2txt(self, save_name):  # excel数据转换为txt
        self.excel_print()
        filenames = save_name
        fs_list = []
        try:
            fs_list.append(open(filenames, 'w', encoding='utf-8'))
            for line in self.data_read.splitlines():
                fs_list[0].write(Function.new_string(line, '@') + '\n')
        except IOError as ex:
            print(ex)
            print('写文件时发生错误~~~~')

        finally:
            for fs in fs_list:
                fs.close()
        print('操作完成~~~~~~~~')

    def excel_data2temp_files(self):
        ss = pd.read_excel(self.file_path)  # 读取数据,设置None可以生成一个字典，字典中的key值即为sheet名字，此时不用使用DataFram，会报错
        # print(ss.index)  # 获取行的索引名称
        # print(ss.columns)  # 获取列的索引名称
        # print(ss)
        # 获取总和、得出行数和列数
        ss_count = ss.shape
        line = ss_count[0]
        row = ss_count[1]
        print('ss_count = {}, 数据记录条数：行={}, 列={}'.format(ss_count, line, row))

        aa = ''
        temp_i = 1
        row_selection = row_return_list(self.file_path)
        print(row_selection, '~~~~~~~~~~~')

        row_selection.insert(0, 0)

        for j_line in range(line):  # j为行
            temp_i += 1
            # print('当前行为：', ss.loc[j_line])
            print('#', temp_i, '标签：', ss.loc[j_line].iloc[0])

            # for i in range(0, row):   #全列读取
            #     #         print(ss.loc[j_line].ix[i], end='')
            #     # aa += str(ss.loc[j_line].iloc[i]) + '@'
            #     aa += str(ss.loc[j_line].iloc[i]).replace('\n', '') + '@'
            #     # 主要修改，！！！.replace('\n', '')增加对excel每行每个cell中换行情况的处理！！！！！！！！！

            # 选取部分列读取：：：：：

            for i in row_selection:   # 选取部分列读取
                #         print(ss.loc[j_line].ix[i], end='')
                # aa += str(ss.loc[j_line].iloc[i]) + '@'
                aa += str(ss.loc[j_line].iloc[i]).replace('\n', '') + '@'
                # 主要修改，！！！.replace('\n', '')增加对excel每行每个cell中换行情况的处理！！！！！！！！！
            aa += '\n'
        print(type(aa), self.file_path)
        temp_i = 1
        for line in aa.splitlines():  # 对字符串按行读取
            temp_i += 1
            print('#', temp_i, ':', line)
        # print(aa,'~~~~~~~~~~~~~~~~')
        self.data_read_files = aa  # 生成读取数据，为后续操作准备

    # def excel2files(self, save_path):
    def excel2different_files(self):
        # filenames = save_path
        # fs_list = []
        try:
            # fs_list.append(open(filenames, 'w', encoding='utf-8'))
            for line_temp in self.data_read_files.splitlines():
                # print(line, '======')
                # fs_list[0].write(Function.new_string(line, '@') + '\n')
                # print(line_temp, '!!!!!!!!!!')
                # line = line_temp.replace('\n', '')
                line_new = Function.new_string(line_temp, '@')
                print(line_new, '输入按照@分开结果')
                # with open(file_names, 'w+', encoding='utf-8') as w:
                #     for aa in line_new:
                aa = line_new
                aa = aa.lower()
                for word in aa:
                    # print(word)
                    if str(word) in stop_words:
                        aa = aa.replace(word, ' ')
                print(aa, '小写--停用词结果')
                #         w.write(aa)
                aa_label = aa.split()[0].replace('/', '')  # 替换标签里面 '/'
                if aa_label == '电池电池配件':
                    aa_label = '电池配件'
                # print(aa_label)
                if aa_label != 'nan':
                    aa_description = " ".join(aa.split()[1:])
                    aa_description = aa_description.replace('nan', '')
                    # ///////////////// 写入对应的 txt文件
                    #                 merge_path + '\\' + name0
                    target_path = 'data\excel_write'
                    target_path = target_path + '\\' + aa_label + '.txt'

                    jj = self.txt_write_line(target_path, aa_description)
                    if jj == False:
                        return self.file_path



                    # //////////////////// line.replace('nan', '')
                    # print('=========', aa_label)
                    # fs_list[0].write(aa_description + '\n')
                    # print('原始excel行：', line_new)
        except IOError as ex:
            print(ex)
            print('写文件时发生错误!!!!!!')#\033[1;31m 字体颜色：红色\033[0m
            return self.file_path
        # finally:
        #     for fs in fs_list:
        #         fs.close()
        print('操作完成!')



if __name__ == '__main__':
    tag_excel = 1
    tag_txt = 100
    if tag_excel == 1:
        # # a = Operate_excel(r'D:\dufy\code\2019-11-25\test.xls')
        # # a = Operate_excel(r'D:\dufy\业务数据\download_bom\2c93ea3b6dedcd55016df63ed1c60457-U137.xls')
        # a = Operate_excel(r'D:\dufy\业务数据\aaa.xls')
        # # a.excel_print()
        # # a.excel2txt('excel_turn_initial.txt')
        # a.excel_data2different_files()
        # a.excel2files(r'data\excel_write\aa.txt')
        filePath = r'D:\dufy\code\ft_BOM\data\bom_test'  # 读取文件夹路径
        file_names = os.listdir(filePath)
        error_path = []
        for i, name0 in enumerate(file_names):    # 文件夹下文件循环
            print('==========================')
            # print(i, name0, type(name0))
            path = filePath + '\\' + name0
            print('path为： ', path)

            row_selection = row_return_list(path)
            if row_selection == []:
                continue
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            aa = Operate_excel(path)
            aa.excel_data2temp_files()  # 生成temp @文件，为后续处理做准备
            # aa.excel2files(r'data\excel_write\aa.txt')  #读取当前excel覆盖写入
            err = aa.excel2different_files()  #读取当前excel覆盖写入
            if err != None:
                error_path.append(err)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            print('path为： ', path)
            print('==========================')
        print(error_path)
        for i in error_path:
            print(i)  # 打印错误路径
        # b = Fasttext_input_norm('excel_turn_terminal.txt', r'D:\dufy\code\2019-11-25\excel_turn_initial.txt')
        # # Fasttext_input_norm(保存路径, 读取路径)
        # b.write_txt()

    if tag_txt == 1:
        # txt操作：：：# Fasttext_input_norm(保存路径, 读取路径)
        # c = Fasttext_input_norm('txt_turn_terminal.txt', r'D:\dufy\code\2019-11-25\test.txt')
        c = Fasttext_input_norm('txt_turn_terminal.txt', r'D:\dufy\code\2019-11-25\data\原始txt数据\光电器件.txt')
        c.write_txt()
        print('=========')
        # 通过for-in循环逐行读取
        with open('txt_turn_terminal.txt', mode='r', encoding='utf-8') as f:
            i = 0
            for line in f:
                i += 1
                print('\033[1;32m #{}\033[0m'.format(i), line, end='')  # end=' '意思是末尾不换行，加空格。
                # time.sleep(0.1)




