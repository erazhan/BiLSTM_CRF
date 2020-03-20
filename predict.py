import tensorflow as tf
import numpy as np

from model import BiLSTMCRF
from predata import open_file
import tfa_crf


'''
#标准化原始的标签列表，调整出现不合理的顺序
#不能出现:
#(b,s),(b,b)
#(m,b),(m,s)
#(e,m),(e,e)
#(s,e),(s,m)
'''
def tag_finetune(label_list):
    for i in range(len(label_list)-1):
        label,next_label=label_list[i],label_list[i+1]
        
        #出现('b','s')或('b','b')
        if label=='b' and (next_label=='s' or next_label=='b'):
            label_list[i]='s'

        #出现('m','s')或('m','b')
        if label=='m' and (next_label=='s' or next_label=='b'):
            if i==0:
                label_list[i]='s'
            else:
                label_list[i]='e'

        #出现('e','m')或('e','e')
        if label=='e' and (next_label=='m' or next_label=='e'):
            if i==0:
                label_list[i]='b'
            else:
                #保险起见将后面的改为's'
                label_list[i+1]='s'

        #出现('s','m')或('s','e')
        if label=='s' and (next_label=='m' or next_label=='e'):
            label_list[i+1]='s'

        #结尾出现'm'或'b'
        if i==len(label_list)-2 and next_label=='m':
            label_list[i+1]='e'
        if i==len(label_list)-2 and next_label=='b':
            label_list[i+1]='s'
            
    return label_list

#对标准化之后的标签列表进行单词合并
def seg_text(text,label_list):
    
    char_list=[char for char in text]
    
    assert len(char_list)==len(label_list),'字数和标签应该相等'
    final=[]
    
    for char,label in zip(char_list,label_list):
        if label=='s':
            final.append(char)
        elif label=='b':
            word=[char]
        else:
            word.append(char)
            if label=='e':
                final.append(''.join(word))
    
    for i,word in enumerate(final):
        print(i+1,word)

#对一条句子分词预测
def single_predict():

    vocab_size=4688
    embed_size=128
    units=64
    num_tags=4
    
    _,_,char_index_dict,index_char_dict=open_file("./data/data.txt")

    optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.1)
    
    my_model = BiLSTMCRF(vocab_size,embed_size,units,num_tags)
    ckpt = tf.train.Checkpoint(optimizer=optimizer,my_model=my_model)
    ckpt.restore(tf.train.latest_checkpoint("./save_checkpoint/"))

    text=input_text()

    char_index_list=[char_index_dict.get(char,0) for char in text]
    
    text_list=[char for char in text]
    tag_list=['b','m','e','s']
    
    inputs = tf.keras.preprocessing.sequence.pad_sequences([char_index_list], padding='post')

    #predict得到numpy矩阵
    logits,inputs_length=my_model.predict(inputs)

    #viterbi_decode得到最优路径
    path,_=tfa_crf.viterbi_decode(logits[0],my_model.transition_params)

    path_list=[tag_list[index] for index in path]
    new_path_list=tag_finetune(path_list)

    #衡量标签路径更改的程度
    print("标签正常率%.2f%%"%(100*sum([i1==i2 for i1,i2 in zip(path_list,new_path_list)])/len(path_list)))
    
    seg_text(text,new_path_list)


def input_text():
    
    #text="我也想过过过过过过过的生活"
    #text="我们中出了个叛徒"
    #text="你好我的名字是汤姆"
    #text="成功入侵民主党的电脑系统"
    #text="中华人民共和国主席江泽民"
    #text="同胞们朋友们女士们先生们"
    #text="实现祖国的完全统一，是海内外全体中国人的共同心愿。通过中葡双方的合作和努力，按照“一国两制”方针和澳门《基本法》，１９９９年１２月澳门的回归一定能够顺利实现。" 
    #text="3月18日消息，互联网内容平台知乎宣布，截至今年2月底，知乎付费用户数比去年同期增长4倍。知乎同时宣布推出“全民阅读计划”，3月18日至25日，免费开放上千本“盐选”专家讲书，为用户提供更多优质内容和服务。"
    text="中共中央政治局常务委员会3月18日召开会议，分析国内外新冠肺炎疫情防控和经济形势，研究部署统筹抓好疫情防控和经济社会发展重点工作。中共中央总书记习近平主持会议并发表重要讲话。"

    return text
if __name__=="__main__":

    single_predict()
    

    
    
