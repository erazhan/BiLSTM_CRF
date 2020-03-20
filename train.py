import tensorflow as tf
import numpy as np
import time

import tfa_crf
from model import BiLSTMCRF
from predata import generate_batch_data,save_file

#@tf.function
def train_one_step(inputs,targets):
    with tf.GradientTape() as tape:
        
        logits,inputs_length=my_model(inputs)

        #参数用来计算loss和train_acc
        log_likelihood,transition_params=my_model.loss(logits,targets,inputs_length)
        
        loss=-tf.math.reduce_sum(log_likelihood)
        
        #trainable_variables
        gradients = tape.gradient(loss, my_model.trainable_variables)
        
        # 反向传播，自动微分计算
        optimizer.apply_gradients(zip(gradients, my_model.trainable_variables))

        # 记录loss和acc，
        train_loss(loss)

        #仅bilstm的预测精度
        lstm_acc(targets,logits)
    return logits,inputs_length

#定义bilstm+crf的预测精度
def get_acc_one_step(logits, text_lens, labels_batch):
    paths = []
    accuracy = 0
    for logit, text_len, labels in zip(logits, text_lens, labels_batch):

        #viterbi_path, _ = tfa.text.viterbi_decode(logit[:text_len], model.transition_params)
        viterbi_path, _ = tfa_crf.viterbi_decode(logit[:text_len], my_model.transition_params)
        paths.append(viterbi_path)
        correct_prediction = tf.equal(
            tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([viterbi_path], padding='post'),
                                 dtype=tf.int32),
            tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([labels[:text_len]], padding='post'),
                                 dtype=tf.int32)
        )
        accuracy = accuracy + tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # print(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
    accuracy = accuracy / len(paths)
    return accuracy

if __name__=="__main__":
    
    #总共有num_inputs=19484条数据,
    #设置batch_size=64
    #则有step=num_inputs/batch_size=300
    #一共有char_size=4688

    batch_size=64

    #单字个数，包括数据集中所有出现的字符
    vocab_size=4688

    #词嵌入维度
    embed_size=128

    #lstm结构中隐藏层维度
    units=64

    #标签类别
    tag_list=['b','m','e','s']
    num_tags=len(tag_list)

    datafile="./data/data.txt"
    
    my_model=BiLSTMCRF(vocab_size,embed_size,units,num_tags)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    lstm_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
    #optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    #optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.1)
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.1)
    #优化器大致效果Adagrad>Adam>RMSprop>SGD

    #设置checkpoint，只保存最新的3个
    ckpt = tf.train.Checkpoint(my_model=my_model,optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt,'./save_checkpoint/',max_to_keep=3)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Load last checkpoint restore')

    for epoch in range(10):
        
        # 重置
        train_loss.reset_states()
        lstm_acc.reset_states()
        
        for step, (inputs, targets) in enumerate(generate_batch_data(datafile,batch_size)):
            
            logits,inputs_length=train_one_step(inputs, targets)

            #损失会比较大，主要看acc
            
            if (step+1) % 100 == 0:
                crf_acc=get_acc_one_step(logits, inputs_length, targets)
                print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
                print('epoch:{},step:{}, loss:{:.4f},lstm_acc:{:.4f},crf_acc:{:.4f}'.format(epoch+1, step+1, train_loss.result(), lstm_acc.result(), crf_acc))
                
                #保存checkpoint
                ckpt_manager.save()
            
