import csv
from os import listdir
from os.path import isfile, join
import random
import pandas as pd


root_dir = '/home/dlevy/PycharmProjects/nerf_pl/datasets/michmoret/dense/images'

onlyfiles = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]
ids =  random.sample(range(10, 70000), len(onlyfiles))
train_test = ['train']*len(onlyfiles)
index_test = [0,8,16]
replacement = ['test']*len(index_test)
for (index, replacement) in zip(index_test, replacement):
    train_test[index] = replacement
name_dict = {'filename': onlyfiles,
            'id': ids,
            'split' :train_test,
            'dataset':['michmoret']*len(onlyfiles)
          }

df = pd.DataFrame(name_dict,)

df.to_csv('michmoret.tsv',sep='\t',index=False)




with open('test.csv', 'w', newline='') as myfile:
    fieldnames = ['filename', 'id', 'split' ,'dataset']
    writer = csv.DictWriter(myfile, fieldnames=fieldnames )
    writer.writeheader()
    writer.writerow({'filename': 'ani.jpg', 'id':'1212','split':'train','dataset':'katzaa'})
    # wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    # wr.writerow(onlyfiles)