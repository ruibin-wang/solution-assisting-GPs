import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

train_data = pd.read_csv("..//data//bert_predict//primary_only_dataset//train.csv")
test_data = pd.read_csv("..//data//bert_predict//primary_only_dataset//test.csv")
val_data = pd.read_csv("..//data//bert_predict//primary_only_dataset//val.csv")


orignal_data = pd.concat([train_data, test_data, val_data], ignore_index=True)

orignal_data.to_csv("..//data//bert_predict//primary_only_dataset//orignal_data.csv")

x = list(orignal_data["diagnosis"])

counter = dict(Counter(x))

# labels_5 = {"A":0, "B":1, "C":2, "D":3, "E":4, "F":5, "G":6, "H":7, "I":8, "J":9, "K":10, "L":11, "M":12, "N":13, "O":14, "P":15, "Q":16}

labels =  ['I60.9', 'G35', 'R55', 'R56.9', 'G43.9',  'G61.0', 'R29.8',  'G40.3',  'R41.0',  'A41.9', 'R51',  'G40.9',  'N39.0',  'I63.9',  'J69.0',  'M54.5',  'J18.9']
values = [12, 34, 36, 87,35,17,24,18,11,16,50,20,15,30,12, 9,13]

import xlwt

# 设置Excel编码
file = xlwt.Workbook('encoding = utf-8')

# 创建sheet工作表
sheet1 = file.add_sheet('sheet1', cell_overwrite_ok=True)

# 先填标题
# sheet1.write(a,b,c) 函数中参数a、b、c分别对应行数、列数、单元格内容
sheet1.write(0, 0, "Disease Code")  # 第1行第1列
sheet1.write(0, 1, "Number of Disease")  # 第1行第2列


# 循环填入数据
for i in range(len(labels)):
    sheet1.write(i + 1, 0, labels[i])  # 第2列数量
    sheet1.write(i + 1, 1, values[i])  # 第3列误差

# 保存Excel到.py源文件同级目录
file.save('E:\RuibinFiles\Pubilshed Papers\LLM\\bar.xls')

# 数据
# labels = ['A', 'B', 'C', 'D', 'E']
# values = [5, 7, 3, 8, 6]

# 创建柱状图
fig, ax = plt.subplots()
bars = ax.bar(labels, values, color='skyblue')

# plt.bar(labels, values)

# 在柱子里面添加数字
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval/2, str(yval),
            ha='center', va='bottom', color='black', fontsize=10)

# 设置标题和标签
# ax.set_title('Distribution of cases of each disease')
plt.xticks(rotation=90)
plt.tight_layout()

ax.set_xlabel('Disease Code')
ax.set_ylabel('Number of Disease')

plt.show()


# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # 示例数据
# data = {
#     '标签': ['A', 'B', 'C', 'D', 'E'],
#     '值': [5, 3, 6, 7, 2]
# }
# df = pd.DataFrame(data)
#
# # 绘制柱状图
# plt.figure(figsize=(10, 6))
# ax = sns.barplot(x='标签', y='值', data=df, palette='viridis')
#
# # 在柱子里面添加数字
# for p in ax.patches:
#     ax.annotate('{:.1f}'.format(p.get_height()),
#                 (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha='center',
#                 va='center',
#                 xytext=(0, 10),
#                 textcoords='offset points')
#
# plt.title('示例柱状图')
# plt.tight_layout()
# plt.show()

# print(11)
