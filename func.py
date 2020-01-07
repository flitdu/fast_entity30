# -*- coding: utf-8 -*-

"""
-------------------------------------------------
Created by Dufy on 2019/11/25
IDE used: PyCharm Community Edition
Description :
1)定义经常用到的函数
2)
-------------------------------------------------
Change Activity:

-------------------------------------------------
"""

import fastText.FastText as ff
import jieba
jieba.load_userdict('dict_boom.txt')

# classfier = ff.load_model("Model/model_w1_e23")
def txt_write_line(save_path, line):  # 将某一行写入txt
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
        # print(self.file_path)
        return False

def load_stop_word_list(file_path):
    """
    加载停用词表
    :param file_path: 停用词表路径
    :return:
    """
    stop_words = set()
    with open(file_path, "r", encoding="utf-8") as f_stopwords:
        for line in f_stopwords:
            stop_words.add(line.strip())
    return stop_words



# def new_string(str1,spilt_str=None):  # 分词规范化, ''__label__电阻	精密贴片电阻	1KΩ ±1%   0603	   5000--->>''__label__电阻 精密贴片电阻 1KΩ ±1% 0603 5000
def string_split_combine(str1,spilt_str=None):  # 分词规范化, ''__label__电阻	精密贴片电阻	1KΩ ±1%   0603	   5000--->>''__label__电阻 精密贴片电阻 1KΩ ±1% 0603 5000
    new_str = ''
    new_list = str1.split(spilt_str)
    for i in new_list:
        new_str += i + ' '
    return new_str.lstrip()

# def sort_test(str1):


# def standard(str1, model_path):
def standard(str1):
    # 加载停用词表
    stop_words = load_stop_word_list("stopwords.txt")
    # classfier = ff.load_model(model_path)
    # print(model_path,'--------')
    # label_to_name = load_label_name_map()[0]
    # classifier = ff.load_model("Model/model_w1_e177")   #
    # classifier = ff.load_model("Model/model_w1_e48")
    # classifier = ff.load_model("Model/model_w1_e223")

    aa_description = str1.lower()
    aa_description = aa_description.replace('nan', '')

    aa_description = ' '.join(jieba.cut(str(aa_description).lower()))  # 先jieba 分词
    print('jieba 分词后输入：{}'.format(aa_description))
    for word in aa_description.split():  # 停用词使用
        # print(word)
        if str(word) in stop_words:
            aa_description = aa_description.replace(word, '')
    aa_description = ' '.join(filter(lambda x: x, aa_description.split(' ')))
    # for word in input_:
    #     # print(word)
    #     if str(word) in stop_words:
    #         input_ = input_.replace(word, ' ')
    print('excel行最终输入：{}'.format(aa_description))
    return aa_description