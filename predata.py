import tensorflow as tf
import numpy as np
import re
import os
import pickle
from collections import Counter

#打开保存的文件
def open_file(fname):
    with open(fname,'rb') as fr:
        ans=pickle.load(fr)
    return ans

#保存处理好的文件
def save_file(fname,data):
    with open(fname,'wb') as fr:
        pickle.dump(data,fr)

#将词语中的每个字进行标注label,仅考虑了中文分词问题，tag_list=['s','b','m','e']
def mark_word(word):
    assert len(word)>0,"输入词是空字符"
    
    if len(word)==1:
        labels=['s']
    elif len(word)==2:
        labels=['b','e']
    else:
        labels=['m']*len(word)
        labels[0],labels[-1]='b','e'
    return labels

#处理原始数据集
def get_data(filename):
    X,Y=[],[]
    char_dict=Counter()
    with open(filename,encoding='utf-8') as f:

        #可以手动设定max_seq_length，超过的截断或删除
        for c,line in enumerate(f.readlines()):
            L=line.strip().split()[1:]
            if len(L)==0:continue
            word_list=[re.sub('^\[','',word_pos.split('/')[0]) for word_pos in L]
            X.append(word_list)
            y_list=[]
            
            for word in word_list:
                char_dict.update(word)
                y_list.append(mark_word(word))
             
            Y.append(y_list)
    
    char_list=[char for char in char_dict.keys()]
    char_list.insert(0,'pad')
    char_index_dict={k:i for i,k in enumerate(char_list)}
    index_char_dict={v:k for k,v in char_index_dict.items()}
    print(len(char_list))
    #一共4688个单字字符(包括pad)
    
    return X,Y,char_index_dict,index_char_dict

#生成每条数据的迭代器
def generate_data(datafile):
    
    X,Y,char_index_dict,index_char_dict=open_file(datafile)
    
    #tag_list=['tag_pad','b','m','e','s']
    tag_list=['b','m','e','s']
    #若由from_generator进行传输会转为numpy数组,而且字符串变成二进制类型，当然也可以decode
    #因为列表不大，为方便直接内部给定
    
    for x,y in zip(X,Y):
        assert len(x)==len(y),"句子字数和标记长度不相等"
        x_vec,y_vec=[],[]
        
        for i in range(len(x)):
            x_vec+=[char_index_dict[char] for char in x[i]]
            y_vec+=[tag_list.index(tag) for tag in y[i]]
            
        yield (x_vec,y_vec)

#构建tf.data.Dataset,在train.py中调用
def generate_batch_data(datafile,batch_size):
    '''可根据需要调整数据重复次数，和预加载batch个数'''
    
    num_repeat,num_prefetch=1,1

    output_types=(tf.int32, tf.int32)
    output_shapes=([None],[None])
    paddings = (0,0)
    
    
    dataset=tf.data.Dataset.from_generator(generate_data,output_types=output_types,output_shapes=output_shapes,args=[datafile])
    #args中的每个参数都会经过numpy转换类型，特别注意会将字符类型转为二进制字符类型，需要额外解码
    
    dataset = dataset.repeat(num_repeat)
    
    dataset = dataset.padded_batch(batch_size, output_shapes, paddings).prefetch(num_prefetch)
    
    return dataset

#将处理好的数据直接保存起来
def save_data():
    #数据是人民日报199801
    raw_path="./data/rawdata.txt"
    save_path="./data/data.txt"
    
    result=get_data(raw_path)
    save_file(save_path,result)

if __name__=="__main__":
    #训练之前需要执行predata.py得到./data/data.txt
    save_data()
