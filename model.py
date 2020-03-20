import tensorflow as tf

#import tensorflow_addons as tfa
import tfa_crf
#tfa_crf文件是tfa.text中关于crf的一个文件
#可直接拿来用，不用安装整个tfa

from tensorflow.keras.layers import Layer,LSTM,Bidirectional,Embedding,Input,Dense,Dropout
from tensorflow.keras.initializers import glorot_uniform
import numpy as np

'''
参数总览：
inputs：维度是[batch_size,max_seq_length,embeded_size]，最简单方法是用ont-hot，
        但是怕单字非常多，可以考虑UNK，取出现次数较多的前N个单字，后面的用UNK代替
units：LSTM结构中h和c的大小
注意要加上每个batch中的长度，用于pad和mask
'''

class BiLSTMCRF(tf.keras.models.Model):
    '''vocab_size,单字个数，embed_size,词向量维度，units，隐藏层维度'''
    
    def __init__(self,vocab_size,embed_size,units,num_tags,*args,**kwargs):
        
        super(BiLSTMCRF,self).__init__()
        self.num_tags=num_tags
        self.embedding=Embedding(input_dim=vocab_size,output_dim=embed_size)
        self.bilstm=Bidirectional(LSTM(units,return_sequences=True),merge_mode='concat')
        #merge_mode的选择从维度角度是不影响输出结果的
        self.dense=Dense(num_tags)
        self.dropout=Dropout(0.5)
        
    def call(self,inputs):
        '''inputs维度：[batch_size,max_seq_length]'''
        inputs_length=tf.math.reduce_sum(tf.cast(tf.math.not_equal(inputs,0),dtype=tf.int32),axis=-1)
        #自动计算每个batch的seq_length，注意数据处理时pad=0
        
        x=self.embedding(inputs)
        x=self.bilstm(x)
        
        x=self.dropout(x)
        logits=self.dense(x)
        
        return logits,inputs_length
        
    def loss(self,logits,targets,inputs_length):
        targets=tf.cast(targets,dtype=tf.int32)

        #计算对数似然函数
        #log_likelihood,_=tfa.text.crf_log_likelihood(logits,targets,inputs_length,transition_params=self.transition_params)
        log_likelihood,_=tfa_crf.crf_log_likelihood(logits,targets,inputs_length,transition_params=self.transition_params)
        
        return log_likelihood,self.transition_params

    #定义转移矩阵transition_params
    def build(self,input_shape):
        shape=tf.TensorShape([self.num_tags,self.num_tags])
        self.transition_params=self.add_weight(name='transition_params',shape=shape,initializer=glorot_uniform,trainable=True)
        super(BiLSTMCRF,self).build(input_shape)

def test():
    batch_size=16
    vocab_size=100
    max_seq_length=8
    embed_size=32
    units=13
    num_tags=50
    
    inputs=tf.convert_to_tensor(np.random.randint(0,100,[batch_size,max_seq_length]),dtype=tf.int32)
    targets=tf.convert_to_tensor(np.random.randint(0,num_tags,[batch_size,max_seq_length]),dtype=tf.int32)
    my_model=BiLSTMCRF(vocab_size,embed_size,units,num_tags)
    
    logits,inputs_length=my_model(inputs)
    
    log_likelihood,transition_params=my_model.loss(logits,targets,inputs_length)

    #print(my_model.transition_params)
    return log_likelihood,transition_params
    
    
if __name__=="__main__":
    
    ll,tp=test()
    print(ll.shape)
    print(tp.shape)
