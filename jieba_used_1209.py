# -*- coding: utf-8 -*-
"""
Created by Dufy on 2019/12/9  10:01
IDE used: PyCharm 
Description :
1)target_path = 'data\excel_write'
2)  
Remark:  注意target_path = 'data\excel_write'要与   txt_filePath = r'D:\dufy\code\ft_BOM\data\excel_write'  # 读取文件夹路径,
         保持一致
"""
import os
import jieba
jieba.load_userdict('dict_boom.txt')
import pandas as pd
from func import string_split_combine, load_stop_word_list
stop_words = load_stop_word_list("stopwords.txt")

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
            # return False

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
                print('excel矩阵单元格：',ss.loc[j_line].ix[i], end='')
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
                fs_list[0].write(string_split_combine(line, '@') + '\n')
        except IOError as ex:
            print(ex)
            print('写文件时发生错误~~~~')

        finally:
            for fs in fs_list:
                fs.close()
        print('操作完成~~~~~~~~')

    def excel_data2temp_files(self):
        ss = pd.read_excel(self.file_path)  # 读取数据,设置None可以生成一个字典，字典中的key值即为sheet名字，此时不用使用DataFram，会报错
        ss_count = ss.shape
        line = ss_count[0]
        row = ss_count[1]
        print('ss_count = {}, 数据记录条数：行={}, 列={}'.format(ss_count, line, row))

        aa = ''
        temp_i = 1
        # row_selection = row_return_list(self.file_path)
        # print(row_selection, '~~~~~~~~~~~')
        #
        # row_selection.insert(0, 0)

        for j_line in range(line):  # j为行
            temp_i += 1
            # print('当前行为：', ss.loc[j_line])
            print('#', temp_i, '标签：', ss.loc[j_line].iloc[0])

            for i in range(0, row):   #全列读取
                #         print(ss.loc[j_line].ix[i], end='')
                # aa += str(ss.loc[j_line].iloc[i]) + '@'
                # print('excel矩阵单元格：', ss.loc[j_line].ix[i], end='')
                excel_cell = str(ss.loc[j_line].iloc[i])
                # if excel_cell != 'nan':  # 替换nan 情形
                # excel_cell = ' '.join(jieba.cut(excel_cell))
                    # 替换
                    # aa += str(ss.loc[j_line].iloc[i]).replace('\n', '') + '@'
                aa += excel_cell.replace('\n', '') + '@'
            # print(aa,type(aa),'!!!!!!!!!!!')
            aa += '\n'
        # print(aa, '~~~~~~~~~~~~~~')
        temp_i = 1
        for line in aa.splitlines():  # 对字符串按行读取
            temp_i += 1
            # print('#', temp_i, ':', line)
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
                line_new = string_split_combine(line_temp, '@')
                # print(line_new, '输入按照@分开结果，jieba 分词后')
                # with open(file_names, 'w+', encoding='utf-8') as w:
                #     for aa in line_new:
                aa = line_new
                # for word in aa.split():  #停用词使用
                #     # print(word)
                #     if str(word) in stop_words:
                #         aa = aa.replace(word, ' ')
                # print(aa, '小写--停用词结果')  #移到后面
                aa_label = aa.split()[0].replace('/', '')  # 替换标签里面 '/'
                # aa_label = aa.split()[0] # 替换标签里面 '/'
                if aa_label == '电池电池配件':
                    aa_label = '电池配件'
                if aa_label == '功能模块开发板方案验证板':
                    aa_label = '方案验证板'
                if aa_label == '二级管':
                    aa_label = '二极管'
                if aa_label == '仪器仪表及配件':
                    aa_label = '仪器仪表'
                if aa_label == '天线':
                    aa_label = '射频无线电'
                if aa_label == '光耦':
                    aa_label = '光电器件'
                if aa_label == '处理器和控制器':
                    aa_label = '处理器和微控制器'
                if aa_label == '险丝座':
                    aa_label = '保险丝'
                if aa_label == '模拟开关':
                    aa_label = '模拟芯片'
                if aa_label == '逻辑器件':
                    aa_label = '逻辑芯片'

                # print(aa_label)
                if aa_label != 'nan':
                    aa_description = " ".join(aa.split()[1:])
                    aa_description = aa_description.replace('nan', '')

                    aa_description = ' '.join(jieba.cut(str(aa_description).lower()))
                    for word in aa_description.split():  # 停用词使用
                        # print(word, '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                        if str(word) in stop_words:
                            aa_description = aa_description.replace(word, '')
                    aa_description = ' '.join(filter(lambda x: x, aa_description.split(' ')))
                    print('最终写入行为：{}'.format(aa_description))
                    aa_description_length = 0
                    for i in aa_description.split(' '):
                        if i != '':
                            aa_description_length += 1
                    # print(length)

                    # Function.new_string(line
                    # ///////////////// 写入对应的 txt文件
                    #                 merge_path + '\\' + name0
                    target_path = 'data\excel_write'
                    target_path = target_path + '\\' + aa_label + '.txt'
                    if aa_description_length > 1:  # 选取训练数据的长度，大于3才算
                        self.txt_write_line(target_path, aa_description)
                    # if jj == False:
                    #     return self.file_path

                    # //////////////////// line.replace('nan', '')
                    # print('=========', aa_label)
                    # fs_list[0].write(aa_description + '\n')
                    # print('原始excel行：', line_new)
        except IOError as ex:
            print(ex)
            print('写文件时发生错误!!!!!!')#\033[1;31m 字体颜色：红色\033[0m
            # return self.file_path
        # finally:
        #     for fs in fs_list:
        #         fs.close()
        print('操作完成!')

def excel_read2txt():
    # 先清空：
    txt_filePath = r'D:\dufy\code\work_record\data\excel_write'  # 读取文件夹路径,
    txt_names = os.listdir(txt_filePath)
    for i, name0 in enumerate(txt_names):  # 文件夹下文件循环
        path = txt_filePath + '\\' + name0
        f_1 = open(path, 'w')
        f_1.truncate()
        f_1.close()

    # filePath = r'C:\Users\Administrator\Documents\Tencent Files\3007490756\FileRecv\bom_test_random'  # 读取文件夹路径
    filePath = r'D:\dufy\code\ft_BOM\data\bom_test'  # 读取文件夹路径!!!!!!!!!!!!
    file_names = os.listdir(filePath)

    for i, name0 in enumerate(file_names):  # 文件夹下文件循环
        print('==========================')
        path = filePath + '\\' + name0
        print('path为： ', path)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        aa = Operate_excel(path)
        aa.excel_data2temp_files()  # 生成temp @文件，为后续处理做准备
        # # aa.excel2files(r'data\excel_write\aa.txt')  #读取当前excel覆盖写入
        aa.excel2different_files()  # 读取当前excel覆盖写入
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print('path为： ', path)
        print('==========================')

# word1 = jieba.cut('RMK1608KB- 1/16W - 20kΩ - F - GJB1432B晶振         灰色 03.01.04.0030	接插件	400	2EDGK弯头板载单排公	灰色 PIN4D5.0	')
#
# print(type(word1))
# for word in word1:
#     print(word, end=' ')


