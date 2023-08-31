<<<<<<< HEAD
import umap as up
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from load_embedding import main


#### primary
prim_train_data = pd.read_csv("..//data//bert_predict//primary_only_dataset//train.csv")
prim_test_data = pd.read_csv("..//data//bert_predict//primary_only_dataset//test.csv")
prim_val_data = pd.read_csv("..//data//bert_predict//primary_only_dataset//val.csv")

prim_data = pd.concat([prim_train_data, prim_test_data, prim_val_data], ignore_index=True)
prim_data.to_csv("..//data//bert_predict//generation_only_dataset//gen_prim_data.csv")


#### generation
gen_train_data = pd.read_csv("..//data//bert_predict//generation_only_dataset//train.csv")
gen_test_data = pd.read_csv("..//data//bert_predict//generation_only_dataset//test.csv")
gen_val_data = pd.read_csv("..//data//bert_predict//generation_only_dataset//val.csv")


gen_data = pd.concat([gen_train_data, gen_test_data, gen_val_data], ignore_index=True)
gen_data.to_csv("..//data//bert_predict//generation_only_dataset//gen_prim_data.csv")



# diagnosis_choice = list(gen_prim_data["diagnosis"])


prim_model_path = "..//data//bert_predict//primary_only_dataset//primary_data.h5"
prim_data_path = "..//data//bert_predict//primary_only_dataset//orignal_data.csv"

prim_embedding_set = main(prim_model_path, prim_data_path)
prim_embedding = [list(list(i[0])[0]) for i in prim_embedding_set]
prim_x = np.array([i[0] for i in prim_embedding])
prim_y = np.array([i[1] for i in prim_embedding])

plt.scatter(prim_x, prim_y, marker='.', color='tab:blue', alpha=0.7, label= "Collected data embedding")
plt.legend()
# plt.show()


gen_model_path = "..//data//bert_predict//generation_only_dataset//gen_prim_data.h5"
gen_data_path = "..//data//bert_predict//generation_only_dataset//gen_prim_data.csv"

gen_embedding_set = main(gen_model_path, gen_data_path)
gen_embedding = [list(list(i[0])[0]) for i in gen_embedding_set]
gen_x = np.array([i[0] for i in gen_embedding])
gen_y = np.array([i[1] for i in gen_embedding])

plt.scatter(gen_x, gen_y, marker='.', color='tab:orange', alpha=0.2, label= "Augmented data embedding")
plt.legend()
plt.show()





=======
import umap as up
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from load_embedding import main


#### primary
prim_train_data = pd.read_csv("..//data//bert_predict//primary_only_dataset//train.csv")
prim_test_data = pd.read_csv("..//data//bert_predict//primary_only_dataset//test.csv")
prim_val_data = pd.read_csv("..//data//bert_predict//primary_only_dataset//val.csv")

prim_data = pd.concat([prim_train_data, prim_test_data, prim_val_data], ignore_index=True)
prim_data.to_csv("..//data//bert_predict//generation_only_dataset//gen_prim_data.csv")


#### generation
gen_train_data = pd.read_csv("..//data//bert_predict//generation_only_dataset//train.csv")
gen_test_data = pd.read_csv("..//data//bert_predict//generation_only_dataset//test.csv")
gen_val_data = pd.read_csv("..//data//bert_predict//generation_only_dataset//val.csv")


gen_data = pd.concat([gen_train_data, gen_test_data, gen_val_data], ignore_index=True)
gen_data.to_csv("..//data//bert_predict//generation_only_dataset//gen_prim_data.csv")



# diagnosis_choice = list(gen_prim_data["diagnosis"])


prim_model_path = "..//data//bert_predict//primary_only_dataset//primary_data.h5"
prim_data_path = "..//data//bert_predict//primary_only_dataset//orignal_data.csv"

prim_embedding_set = main(prim_model_path, prim_data_path)
prim_embedding = [list(list(i[0])[0]) for i in prim_embedding_set]
prim_x = np.array([i[0] for i in prim_embedding])
prim_y = np.array([i[1] for i in prim_embedding])

plt.scatter(prim_x, prim_y, marker='.', color='tab:blue', alpha=0.7, label= "Collected data embedding")
plt.legend()
# plt.show()


gen_model_path = "..//data//bert_predict//generation_only_dataset//gen_prim_data.h5"
gen_data_path = "..//data//bert_predict//generation_only_dataset//gen_prim_data.csv"

gen_embedding_set = main(gen_model_path, gen_data_path)
gen_embedding = [list(list(i[0])[0]) for i in gen_embedding_set]
gen_x = np.array([i[0] for i in gen_embedding])
gen_y = np.array([i[1] for i in gen_embedding])

plt.scatter(gen_x, gen_y, marker='.', color='tab:orange', alpha=0.2, label= "Augmented data embedding")
plt.legend()
plt.show()





>>>>>>> a39879c (first upload)
print(11)