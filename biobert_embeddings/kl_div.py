<<<<<<< HEAD
import pandas as pd
from load_embedding import main
import numpy as np
import matplotlib.pyplot as plt
import math
from function_sets import read_json, write_to_json
import torch
import torch.nn.functional as F


def draw_embedding_distribution(prim_model_path):

    gen_data_path = "..//data//bert_gen_800//gen_800_cases.csv"
    gen_embedding_set = main(prim_model_path, gen_data_path)
    gen_embedding = [list(list(i[0])[0]) for i in gen_embedding_set]
    gen_x = np.array([i[0] for i in gen_embedding])
    gen_y = np.array([i[1] for i in gen_embedding])

    plt.scatter(gen_x, gen_y, marker='.', color='tab:orange', alpha=0.2, label= "Augmented data embedding")
    plt.legend()

    prim_data_path = "..//data//bert_gen_800//prim_data.csv"

    prim_embedding_set = main(prim_model_path, prim_data_path)
    prim_embedding = [list(list(i[0])[0]) for i in prim_embedding_set]
    prim_x = np.array([i[0] for i in prim_embedding])
    prim_y = np.array([i[1] for i in prim_embedding])

    plt.scatter(prim_x, prim_y, marker='.', color='tab:blue', alpha=0.7, label= "Collected data embedding")
    plt.legend()

    plt.show()

# prim_model_path = "..//data//bert_gen_800//all_collab_dataset.h5"
# draw_embedding_distribution(prim_model_path)



## 字典添加函数
def add_to_dict(d, key, value):
    if key in d:
        if isinstance(d[key], list):
            d[key].append(value)
        else:
            d[key] = [d[key], value]
    else:
        d[key] = value

def cosine_similarity(A, B):
    A = np.array(A)
    B = np.array(B)
    dot_product = np.dot(A, B)
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    return dot_product / (norm_a * norm_b)


gen_data_path = "..//data//bert_gen_800//gen_800_cases.csv"
gen_800_cases = pd.read_csv(gen_data_path)

prim_data_path = "..//data//bert_gen_800//prim_data.csv"
prim_data = pd.read_csv(prim_data_path)


# all_collab_dataset = pd.concat([gen_800_cases, prim_data], ignore_index=True)
# all_collab_dataset.to_csv("..//data//bert_gen_800//all_collab_dataset.csv")

prim_model_path = "..//data//bert_gen_800//all_collab_dataset.h5"

gen_embedding_set = main(prim_model_path, gen_data_path)
gen_embedding = [list(list(i[0])[0]) for i in gen_embedding_set]



prim_embedding_set = main(prim_model_path, prim_data_path)
prim_embedding = [list(list(i[0])[0]) for i in prim_embedding_set]


x = ['A41.9','G35','G40.3','G40.9', 'G43.9', 'G61.0', 'I60.9','I63.9','J18.9', 'J69.0', 'M54.5', 'N39.0', 'R29.8', 'R41.0', 'R51','R55','R56.9']
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q"]

## 将疾病的icd-10 code 和选项标签对应上
disease_label = {}
for i in range(len(labels)):
    disease_label[labels[i]] = x[i]


## 分类，将每种疾病对应的referral letter放到字典中
dict_gen_800_cases = {}
for i in range(0,len(gen_800_cases)):
    add_to_dict(dict_gen_800_cases, disease_label[gen_800_cases["diagnosis"][i]], gen_800_cases["symptoms"][i])


dict_prim_data = {}
for i in range(0,len(prim_data)):
    add_to_dict(dict_prim_data, disease_label[prim_data["diagnosis"][i]], prim_data["symptoms"][i])



# ######################################################
# ## 计算生成data和对应disease下所有data的cosine距离
# disease_cos_dict = {}
# for key in dict_gen_800_cases:
#     cos_list = []
#     for i in dict_gen_800_cases[key]:
#         sum_cos = 0
#         for j in dict_prim_data[key]:
#             sum_cos = sum_cos + cosine_similarity(gen_embedding[gen_800_cases["symptoms"].tolist().index(i)], prim_embedding[prim_data["symptoms"].tolist().index(j)])
#
#         avg_cos = sum_cos/len(dict_prim_data[key])
#         cos_list.append(avg_cos)
#     add_to_dict(disease_cos_dict, key, cos_list)

##########################################################
### 计算KL散度，利用torch中的散度模块

disease_cos_dict = {}
for key in dict_gen_800_cases:
    kl_list = []
    for i in dict_gen_800_cases[key]:

        distrib_collect = [prim_embedding[prim_data["symptoms"].tolist().index(j)] for j in dict_prim_data[key]]
        distrib_gen = [gen_embedding[gen_800_cases["symptoms"].tolist().index(i)] for j in range(0, len(distrib_collect))]

        ten_distrib_collect = torch.Tensor(distrib_collect)
        ten_distrib_gen = torch.Tensor(distrib_gen)

        kl_mean = F.kl_div(F.log_softmax(ten_distrib_collect, dim=-1), F.softmax(ten_distrib_gen, dim=-1), reduction='batchmean')


        kl_list.append(kl_mean.tolist())
    add_to_dict(disease_cos_dict, key, kl_list)


write_to_json("disease_kl_dict.json", disease_cos_dict)

disease_kl_dict = read_json("disease_kl_dict.json")


df = pd.DataFrame.from_dict(disease_cos_dict, orient='index').transpose()
df.to_excel("kl_div.xlsx", index=False, engine='openpyxl')

print(11)


=======
import pandas as pd
from load_embedding import main
import numpy as np
import matplotlib.pyplot as plt
import math
from function_sets import read_json, write_to_json
import torch
import torch.nn.functional as F


def draw_embedding_distribution(prim_model_path):

    gen_data_path = "..//data//bert_gen_800//gen_800_cases.csv"
    gen_embedding_set = main(prim_model_path, gen_data_path)
    gen_embedding = [list(list(i[0])[0]) for i in gen_embedding_set]
    gen_x = np.array([i[0] for i in gen_embedding])
    gen_y = np.array([i[1] for i in gen_embedding])

    plt.scatter(gen_x, gen_y, marker='.', color='tab:orange', alpha=0.2, label= "Augmented data embedding")
    plt.legend()

    prim_data_path = "..//data//bert_gen_800//prim_data.csv"

    prim_embedding_set = main(prim_model_path, prim_data_path)
    prim_embedding = [list(list(i[0])[0]) for i in prim_embedding_set]
    prim_x = np.array([i[0] for i in prim_embedding])
    prim_y = np.array([i[1] for i in prim_embedding])

    plt.scatter(prim_x, prim_y, marker='.', color='tab:blue', alpha=0.7, label= "Collected data embedding")
    plt.legend()

    plt.show()

# prim_model_path = "..//data//bert_gen_800//all_collab_dataset.h5"
# draw_embedding_distribution(prim_model_path)



## 字典添加函数
def add_to_dict(d, key, value):
    if key in d:
        if isinstance(d[key], list):
            d[key].append(value)
        else:
            d[key] = [d[key], value]
    else:
        d[key] = value

def cosine_similarity(A, B):
    A = np.array(A)
    B = np.array(B)
    dot_product = np.dot(A, B)
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    return dot_product / (norm_a * norm_b)


gen_data_path = "..//data//bert_gen_800//gen_800_cases.csv"
gen_800_cases = pd.read_csv(gen_data_path)

prim_data_path = "..//data//bert_gen_800//prim_data.csv"
prim_data = pd.read_csv(prim_data_path)


# all_collab_dataset = pd.concat([gen_800_cases, prim_data], ignore_index=True)
# all_collab_dataset.to_csv("..//data//bert_gen_800//all_collab_dataset.csv")

prim_model_path = "..//data//bert_gen_800//all_collab_dataset.h5"

gen_embedding_set = main(prim_model_path, gen_data_path)
gen_embedding = [list(list(i[0])[0]) for i in gen_embedding_set]



prim_embedding_set = main(prim_model_path, prim_data_path)
prim_embedding = [list(list(i[0])[0]) for i in prim_embedding_set]


x = ['A41.9','G35','G40.3','G40.9', 'G43.9', 'G61.0', 'I60.9','I63.9','J18.9', 'J69.0', 'M54.5', 'N39.0', 'R29.8', 'R41.0', 'R51','R55','R56.9']
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q"]

## 将疾病的icd-10 code 和选项标签对应上
disease_label = {}
for i in range(len(labels)):
    disease_label[labels[i]] = x[i]


## 分类，将每种疾病对应的referral letter放到字典中
dict_gen_800_cases = {}
for i in range(0,len(gen_800_cases)):
    add_to_dict(dict_gen_800_cases, disease_label[gen_800_cases["diagnosis"][i]], gen_800_cases["symptoms"][i])


dict_prim_data = {}
for i in range(0,len(prim_data)):
    add_to_dict(dict_prim_data, disease_label[prim_data["diagnosis"][i]], prim_data["symptoms"][i])



# ######################################################
# ## 计算生成data和对应disease下所有data的cosine距离
# disease_cos_dict = {}
# for key in dict_gen_800_cases:
#     cos_list = []
#     for i in dict_gen_800_cases[key]:
#         sum_cos = 0
#         for j in dict_prim_data[key]:
#             sum_cos = sum_cos + cosine_similarity(gen_embedding[gen_800_cases["symptoms"].tolist().index(i)], prim_embedding[prim_data["symptoms"].tolist().index(j)])
#
#         avg_cos = sum_cos/len(dict_prim_data[key])
#         cos_list.append(avg_cos)
#     add_to_dict(disease_cos_dict, key, cos_list)

##########################################################
### 计算KL散度，利用torch中的散度模块

disease_cos_dict = {}
for key in dict_gen_800_cases:
    kl_list = []
    for i in dict_gen_800_cases[key]:

        distrib_collect = [prim_embedding[prim_data["symptoms"].tolist().index(j)] for j in dict_prim_data[key]]
        distrib_gen = [gen_embedding[gen_800_cases["symptoms"].tolist().index(i)] for j in range(0, len(distrib_collect))]

        ten_distrib_collect = torch.Tensor(distrib_collect)
        ten_distrib_gen = torch.Tensor(distrib_gen)

        kl_mean = F.kl_div(F.log_softmax(ten_distrib_collect, dim=-1), F.softmax(ten_distrib_gen, dim=-1), reduction='batchmean')


        kl_list.append(kl_mean.tolist())
    add_to_dict(disease_cos_dict, key, kl_list)


write_to_json("disease_kl_dict.json", disease_cos_dict)

disease_kl_dict = read_json("disease_kl_dict.json")


df = pd.DataFrame.from_dict(disease_cos_dict, orient='index').transpose()
df.to_excel("kl_div.xlsx", index=False, engine='openpyxl')

print(11)


>>>>>>> a39879c (first upload)
