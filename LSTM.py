# -*- coding: utf-8 -*-

import theano
theano.config.floatX= 'float64'

#out_sc_x is the Parsey McParseface parsing result of original sentence
#out_sc_y is the Parsey McParseface parsing result of summary
###############################################################################
f=open("./out_sc_x.txt",mode="r")

temp=[]
sc_parsed_x=[]
for lines in f.readlines():
    if len(lines.split())!=0:
        temp.append(lines.split())
    else:
        sc_parsed_x.append(temp)
        temp=[]
    
f=open("./out_sc_y.txt",mode="r")

temp=[]
sc_parsed_y=[]
for lines in f.readlines():
    if len(lines.split())!=0:
        temp.append(lines.split())
    else:
        sc_parsed_y.append(temp)
        temp=[]
    

sc_parsed_x1=[]
sc_parsed_y1=[]
for s in sc_parsed_x:
    temp=[]
    for w in s:
        temp.append(w[1])
    sc_parsed_x1.append(temp)
    
for s in sc_parsed_y:
    temp=[]
    for w in s:
        temp.append(w[1])
    sc_parsed_y1.append(temp)


win_nb=5
def win(f, posi, win_nb):
    left=f[max(0,posi-win_nb/2):posi]
    right=f[posi+1:min(len(f),posi+win_nb/2+1)]
    weights=[1+i*0.1 for i in range(win_nb/2)]
    
    left_sum=0
    for i in range(len(left)):
        left_sum+=left[i]*weights[i]
        
    right.reverse()
    right_sum=0
    for i in range(len(right)):
        right_sum+=right[i]*weights[i]
    
    right.reverse()
    
    return left_sum+right_sum
    
    
import collections as col
label_all=[]
for father, son in zip(sc_parsed_x1, sc_parsed_y1):
    f_len=len(father)
    s_len=len(son)
    
    f_mark=[0]*f_len
    s_mark=[0]*s_len
    
    f_col=col.Counter(father)
    #s_col=col.Counter(son)
    
    temp_father=dict()
    temp_son=dict()
    
    for i in range(f_len):
        temp_father[father[i]]=set()      
    
    for i in range(s_len):
        temp_son[son[i]]=set()   
    
    
    for s_id in range(s_len):
        for f_id in range(f_len):
            if son[s_id]==father[f_id]:
                if f_col[son[s_id]]==1:
                    f_mark[f_id]=1
                    s_mark[s_id]=1
                else:
                    if f_col[son[s_id]]>=2:
                        temp_father[son[s_id]].add(f_id)
                        temp_son[son[s_id]].add(s_id)
    
    for ele in temp_father.keys():
        if temp_father[ele]==set():
            del temp_father[ele]
    
    for ele in temp_son.keys():
        if temp_son[ele]==set():
            del temp_son[ele]

    
    for word in temp_son.keys():
        if len(temp_son[word])==len(temp_father[word]):
            for posi in temp_father[word]:
                        f_mark[posi]=1
        else:
            if len(temp_son[word]) < len(temp_father[word]):
                num_max=len(temp_son[word])
                max_context=0
                queue=[]
                for posi in temp_father[word]:
                    queue.append((win(f_mark, posi, win_nb), posi))
                
                queue=sorted(queue, reverse=True)
                for k in range(num_max):
                    f_mark[queue[k][1]]=1
                               
                    
    label_all.append([temp_father,temp_son, f_mark,s_mark])


label=[]
for s in label_all:
    label.append(s[-2])
    
###############################################################################
#train_x: sc_parsed_x1[i]
#train_y: label[i]

print 'Lemmatize...'           
    
change=set()
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()
for i in range(len(sc_parsed_x1)):
    for j in range(len(sc_parsed_x1[i])):
        try:
            temp_a=sc_parsed_x1[i][j]
            sc_parsed_x1[i][j]=lem.lemmatize(sc_parsed_x1[i][j])
            temp_b=sc_parsed_x1[i][j]
            
            if temp_a!=temp_b:
                change.add((i,j,temp_a,temp_b))
        except:
            sc_parsed_x1[i][j]=sc_parsed_x1[i][j].decode("utf-8")
            
change_dict={}


for d in change:
    change_dict[d[:2]]=d[2:]
            
###############################################################################
print "train word2vec using training dataset"+'\n'
import numpy as np
np.random.seed(1337)  # for reproducibility

import multiprocessing
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary


import numpy as np

np.random.seed(1337)  # For Reproducibility


# Embedding
max_features =20000
maxlen = 100


# Convolution
filter_length = 3
nb_filter = 64
pool_length = 2

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
nb_epoch = 2

# set parameters:
emb_dim = 300
n_exposures = 0
window_size = 7
cpu_count = multiprocessing.cpu_count()

#build word2vec(skip-gram) model for dundee corpus
model = Word2Vec(size=emb_dim,
                 min_count=n_exposures,
                 window=window_size,
                 workers=cpu_count,
                 iter=10,
                 sg=1)
                 
model.build_vocab(sc_parsed_x1)
model.train(sc_parsed_x1)

gensim_dict = Dictionary()
gensim_dict.doc2bow(model.vocab.keys(),
                    allow_update=True)
                    
index_dict = {v: k+1 for k, v in gensim_dict.items()}
word_vectors = {word: model[word] for word in index_dict.keys()}
index2word = {index_dict[num]: num for num in index_dict.keys()}

print('Setting up Arrays for Keras Embedding Layer...')
n_symbols = len(index_dict) + 1  # adding 1 to account for 0th index
embedding_weights = np.zeros((n_symbols, emb_dim))
for word, index in index_dict.items():
    embedding_weights[index,:] = word_vectors[word]  

###############################################################################
train_x2 = []
for i in range(len(sc_parsed_x1)):
    temp=[]
    for j in range(len(sc_parsed_x1[i])):
        temp.append(index_dict[sc_parsed_x1[i][j]])
    train_x2.append(temp)


print "theano RNN"

import theano
import theano.tensor as tt
import numpy as np
np.random.seed(1337)
from collections import OrderedDict

def contextwin(l, win):
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)

    lpadded = win // 2 * [0] + l + win // 2 * [0]
    out = [lpadded[i:(i + win)] for i in range(len(l))]

    assert len(out) == len(l)
    return out

nb_voca=30000
win_size = 1
rnn_hdim = 200
rnn_output_dim = 2

x_data = []
for d in train_x2:
    x_data.append(np.array(contextwin(d, win_size),dtype=np.int64))


y_data=[]
for i in range(10000):
    temp=[]
    for j in range(len(label[i])):
        if label[i][j]==1: 
            temp.append([0,1])
        else:
            temp.append([1,0])
    y_data.append(np.array(temp,dtype=np.float32))
        
embedding = theano.shared(embedding_weights.astype(np.float32))
#embedding = theano.shared(np.random.uniform(-1.0, 1.0, (nb_voca+1, emb_dim)).astype(np.float32))
y = tt.fmatrix('y_label')    
idxs = tt.lmatrix('idxs')
x = embedding[idxs].reshape((idxs.shape[0], emb_dim*win_size))


###############################################################################
#the second channal: dependency embeddings
#train_x: sc_parsed_x[i]
#train_y: label[i]
dep=[]
dep_dim=50

for i in range(len(sc_parsed_x)):
    temp=[]
    for j in range(len(sc_parsed_x[i])):
        temp.append(sc_parsed_x[i][j][7])
    dep.append(temp)

dep_all=set()
dep_num=0
for s in dep:
    for d in s:
        dep_all.add(d)
    
dep_all=list(dep_all)
dep_num=len(dep_all)
dep2id=dict()
for i in range(dep_num):
    dep2id[dep_all[i]]=i


dep_idxs=[]
for i in range(len(dep)):
    temp=[]
    for j in range(len(dep[i])):
        temp.append([dep2id[dep[i][j]]])
    dep_idxs.append(np.array(temp))


emb_dep=np.random.uniform(-0.1, 0.1, (dep_num, dep_dim)).astype(np.float32)
embedding_dep = theano.shared(emb_dep.astype(np.float32))

dep_input = tt.lmatrix('dep_input')
x_dep = embedding_dep[dep_input].reshape((dep_input.shape[0], dep_dim))


###############################################################################
#the third channal: pos embeddings
#train_x: sc_parsed_x[i]
#train_y: label[i]
pos=[]
pos_dim=50

for i in range(len(sc_parsed_x)):
    temp=[]
    for j in range(len(sc_parsed_x[i])):
        temp.append(sc_parsed_x[i][j][4])
    pos.append(temp) 

pos_all=set()
pos_num=0
for s in pos:
    for d in s:
        pos_all.add(d)
        
pos_all=list(pos_all)
pos_num=len(pos_all)

pos2id=dict()
for i in range(pos_num):
    pos2id[pos_all[i]]=i

pos_idxs=[]
for i in range(len(pos)):
    temp=[]
    for j in range(len(pos[i])):
        temp.append([pos2id[pos[i][j]]])
    pos_idxs.append(np.array(temp))



emb_pos=np.random.uniform(-0.1, 0.1, (pos_num, pos_dim)).astype(np.float32)
embedding_pos = theano.shared(emb_pos.astype(np.float32))

pos_input = tt.lmatrix('pos_input')
x_pos = embedding_pos[pos_input].reshape((pos_input.shape[0], pos_dim))

###############################################################################
cat_inputs = tt.concatenate([x, x_dep, x_pos], axis=1)

ini_dim=emb_dim+dep_dim+pos_dim


#LSTM computational graph
dtype=theano.config.floatX

n_in = ini_dim # for embedded reber grammar
n_hidden = n_i = n_c = n_o = n_f = 150
n_out = 2 

sigma = lambda x: 1 / (1 + tt.exp(-x))
                
                      
def one_lstm_step1(x_t, h_tm1, c_tm1):

    i_t = sigma(theano.dot(x_t, W_xi1) + theano.dot(h_tm1, W_hi1) + b_i1)
    f_t = sigma(theano.dot(x_t, W_xf1) + theano.dot(h_tm1, W_hf1) + b_f1)
    c_t = f_t * c_tm1 + i_t * tt.tanh(theano.dot(x_t, W_xc1) + theano.dot(h_tm1, W_hc1) + b_c1) 
    o_t = sigma(theano.dot(x_t, W_xo1) + theano.dot(h_tm1, W_ho1) + b_o1)
    h_t = o_t * tt.tanh(c_t)
    #y_t = tt.nnet.softmax(theano.dot(h_t, W_hy1) + b_y1) 
    return [h_t, c_t]
    
def one_lstm_step2(x_t, h_tm1, c_tm1):

    i_t = sigma(theano.dot(x_t, W_xi2) + theano.dot(h_tm1, W_hi2) + b_i2)
    f_t = sigma(theano.dot(x_t, W_xf2) + theano.dot(h_tm1, W_hf2) + b_f2)
    c_t = f_t * c_tm1 + i_t * tt.tanh(theano.dot(x_t, W_xc2) + theano.dot(h_tm1, W_hc2) + b_c2) 
    o_t = sigma(theano.dot(x_t, W_xo2) + theano.dot(h_tm1, W_ho2) + b_o2)
    h_t = o_t * tt.tanh(c_t)
    #y_t = tt.nnet.softmax(theano.dot(h_t, W_hy1) + b_y1) 
    return [h_t, c_t]
    
    
def one_lstm_step3(x_t, h_tm1, c_tm1):

    i_t = sigma(theano.dot(x_t, W_xi3) + theano.dot(h_tm1, W_hi3) + b_i3)
    f_t = sigma(theano.dot(x_t, W_xf3) + theano.dot(h_tm1, W_hf3) + b_f3)
    c_t = f_t * c_tm1 + i_t * tt.tanh(theano.dot(x_t, W_xc3) + theano.dot(h_tm1, W_hc3) + b_c3) 
    o_t = sigma(theano.dot(x_t, W_xo3) + theano.dot(h_tm1, W_ho3) + b_o3)
    h_t = o_t * tt.tanh(c_t)
    #y_t = tt.nnet.softmax(theano.dot(h_t, W_hy1) + b_y1) 
    return [h_t, c_t]
    
def one_lstm_step4(x_t, h_tm1, c_tm1):

    i_t = sigma(theano.dot(x_t, W_xi4) + theano.dot(h_tm1, W_hi4) + b_i4)
    f_t = sigma(theano.dot(x_t, W_xf4) + theano.dot(h_tm1, W_hf4) + b_f4)
    c_t = f_t * c_tm1 + i_t * tt.tanh(theano.dot(x_t, W_xc4) + theano.dot(h_tm1, W_hc4) + b_c4) 
    o_t = sigma(theano.dot(x_t, W_xo4) + theano.dot(h_tm1, W_ho4) + b_o4)
    h_t = o_t * tt.tanh(c_t)
    #y_t = tt.nnet.softmax(theano.dot(h_t, W_hy1) + b_y1) 
    return [h_t, c_t]   


def one_lstm_step5(x_t, h_tm1, c_tm1):

    i_t = sigma(theano.dot(x_t, W_xi5) + theano.dot(h_tm1, W_hi5) + b_i5)
    f_t = sigma(theano.dot(x_t, W_xf5) + theano.dot(h_tm1, W_hf5) + b_f5)
    c_t = f_t * c_tm1 + i_t * tt.tanh(theano.dot(x_t, W_xc5) + theano.dot(h_tm1, W_hc5) + b_c5) 
    o_t = sigma(theano.dot(x_t, W_xo5) + theano.dot(h_tm1, W_ho5) + b_o5)
    h_t = o_t * tt.tanh(c_t)
    #y_t = tt.nnet.softmax(theano.dot(h_t, W_hy1) + b_y1) 
    return [h_t, c_t]
    
    
def one_lstm_step6(x_t, h_tm1, c_tm1):

    i_t = sigma(theano.dot(x_t, W_xi6) + theano.dot(h_tm1, W_hi6) + b_i6)
    f_t = sigma(theano.dot(x_t, W_xf6) + theano.dot(h_tm1, W_hf6) + b_f6)
    c_t = f_t * c_tm1 + i_t * tt.tanh(theano.dot(x_t, W_xc6) + theano.dot(h_tm1, W_hc6) + b_c6) 
    o_t = sigma(theano.dot(x_t, W_xo6) + theano.dot(h_tm1, W_ho6) + b_o6)
    h_t = o_t * tt.tanh(c_t)
    #y_t = tt.nnet.softmax(theano.dot(h_t, W_hy1) + b_y1) 
    return [h_t, c_t]   




def sample_weights(sizeX, sizeY):
    values = np.ndarray([sizeX, sizeY], dtype=dtype)
    for dx in xrange(sizeX):
        vals = np.random.uniform(low=-1., high=1.,  size=(sizeY,))
        values[dx,:] = vals
    _,svs,_ = np.linalg.svd(values)              
    values = values / svs[0]
    return values  


def ortho_weight(n_in, n_hidden, n_i, n_c, n_o, n_f):
    W_xi = theano.shared(sample_weights(n_in, n_i))  
    W_hi = theano.shared(sample_weights(n_hidden, n_i))  
    b_i  = theano.shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_i)))
    
    W_xf = theano.shared(sample_weights(n_in, n_f)) 
    W_hf = theano.shared(sample_weights(n_hidden, n_f))
    b_f  = theano.shared(np.cast[dtype](np.random.uniform(0, 1.,size = n_f)))
    #b_f = theano.shared(np.cast[dtype](np.ones(n_f)))
    
    W_xc = theano.shared(sample_weights(n_in, n_c))  
    W_hc = theano.shared(sample_weights(n_hidden, n_c))
    b_c  = theano.shared(np.zeros(n_c, dtype=dtype))
    
    W_xo = theano.shared(sample_weights(n_in, n_o))
    W_ho = theano.shared(sample_weights(n_hidden, n_o))
    b_o  = theano.shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_o)))
    
    
    c0 = theano.shared(np.zeros(n_hidden, dtype=dtype))
    h0 = tt.tanh(c0)
    
    return W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xc, W_hc, b_c, W_xo, W_ho, b_o, c0, h0

W_xi1, W_hi1, b_i1, W_xf1, W_hf1, b_f1, W_xc1, W_hc1, b_c1, W_xo1, W_ho1, b_o1, c01, h01 = ortho_weight(n_in, n_hidden, n_i, n_c, n_o, n_f)
W_xi2, W_hi2, b_i2, W_xf2, W_hf2, b_f2, W_xc2, W_hc2, b_c2, W_xo2, W_ho2, b_o2, c02, h02 = ortho_weight(n_in, n_hidden, n_i, n_c, n_o, n_f)

W_xi3, W_hi3, b_i3, W_xf3, W_hf3, b_f3, W_xc3, W_hc3, b_c3, W_xo3, W_ho3, b_o3, c03, h03 = ortho_weight(2*n_hidden, n_hidden, n_i, n_c, n_o, n_f)
W_xi4, W_hi4, b_i4, W_xf4, W_hf4, b_f4, W_xc4, W_hc4, b_c4, W_xo4, W_ho4, b_o4, c04, h04 = ortho_weight(2*n_hidden, n_hidden, n_i, n_c, n_o, n_f)

W_xi5, W_hi5, b_i5, W_xf5, W_hf5, b_f5, W_xc5, W_hc5, b_c5, W_xo5, W_ho5, b_o5, c05, h05 = ortho_weight(2*n_hidden, n_hidden, n_i, n_c, n_o, n_f)
W_xi6, W_hi6, b_i6, W_xf6, W_hf6, b_f6, W_xc6, W_hc6, b_c6, W_xo6, W_ho6, b_o6, c06, h06 = ortho_weight(2*n_hidden, n_hidden, n_i, n_c, n_o, n_f)


[h1, _], _ = theano.scan(fn=one_lstm_step1, 
                         sequences = cat_inputs,
                         outputs_info = [h01, c01],
                         n_steps=cat_inputs.shape[0])

[h2, _], _ = theano.scan(fn=one_lstm_step2, 
                         sequences = cat_inputs,
                         outputs_info = [h02, c02],
                         n_steps=cat_inputs.shape[0],
                         go_backwards=True)

hidden12 = tt.concatenate([h1, h2[::-1]], axis=1)



[h3, _], _ = theano.scan(fn=one_lstm_step3, 
                         sequences = hidden12,
                         outputs_info = [h03, c03],
                         n_steps=hidden12.shape[0])
                            
[h4, _], _ = theano.scan(fn=one_lstm_step4, 
                         sequences = hidden12,
                         outputs_info = [h04, c04],
                         n_steps=hidden12.shape[0],
                         go_backwards=True)
                            
hidden34 = tt.concatenate([h3, h4[::-1]], axis=1)



[h5, _], _ = theano.scan(fn=one_lstm_step5, 
                         sequences = hidden34,
                         outputs_info = [h05, c05],
                         n_steps=hidden34.shape[0])
                            
[h6, _], _ = theano.scan(fn=one_lstm_step6, 
                         sequences = hidden34,
                         outputs_info = [h06, c06],
                         n_steps=hidden34.shape[0],
                         go_backwards=True)
                            
hidden56 = tt.concatenate([h5, h6[::-1]], axis=1)



W_hy = theano.shared(sample_weights(2*n_hidden, n_out))#+2*dep_dim
b_y = theano.shared(np.zeros(n_out, dtype=dtype))

softmax = tt.nnet.softmax(tt.dot(hidden56, W_hy) + b_y)  


dep_x_test=dep_idxs[:1000]
pos_x_test=pos_idxs[:1000]
x_test=x_data[:1000]
y_test=label[:1000]


dep_x_train_set = dep_idxs[2000:]
pos_x_train_set = pos_idxs[2000:]
x_train_set = x_data[2000:]
y_train_set = y_data[2000:]


dep_x_val_set = dep_idxs[1000:2000]
pos_x_val_set = pos_idxs[1000:2000]
x_val_set = x_data[1000:2000]
y_val_set = label[1000:2000]


params = [embedding,
          W_xi1, W_hi1, b_i1, 
          W_xf1, W_hf1, b_f1, 
          W_xc1, W_hc1, b_c1, 
          W_xo1, W_ho1, b_o1, 
          c01, 
          
          W_xi2, W_hi2, b_i2, 
          W_xf2, W_hf2, b_f2, 
          W_xc2, W_hc2, b_c2, 
          W_xo2, W_ho2, b_o2, 
          c02, 
          
          W_xi3, W_hi3, b_i3, 
          W_xf3, W_hf3, b_f3, 
          W_xc3, W_hc3, b_c3, 
          W_xo3, W_ho3, b_o3, 
          c03, 
          
          W_xi4, W_hi4, b_i4, 
          W_xf4, W_hf4, b_f4, 
          W_xc4, W_hc4, b_c4, 
          W_xo4, W_ho4, b_o4, 
          c04, 
          
          W_xi5, W_hi5, b_i5, 
          W_xf5, W_hf5, b_f5, 
          W_xc5, W_hc5, b_c5, 
          W_xo5, W_ho5, b_o5, 
          c05, 
          
          W_xi6, W_hi6, b_i6, 
          W_xf6, W_hf6, b_f6, 
          W_xc6, W_hc6, b_c6, 
          W_xo6, W_ho6, b_o6, 
          c06, 
          
          embedding_dep, embedding_pos, 
                         
          W_hy, b_y]
          
###############################################################################
l2_loss=0
for param in params:
    l2_loss+=(param**2).sum()

lamda=theano.shared(np.array(1e-04))

loss1 = tt.mean(tt.nnet.categorical_crossentropy(softmax, y))+ lamda/float(2) * l2_loss


y_pred = tt.argmax(softmax, axis=1)   

gradients = tt.grad(loss1, params)        

lr=tt.fscalar('lr')
updates = OrderedDict(( p, p-lr*g ) 
                      for p, g in zip( params , gradients))        
 
train = theano.function(inputs  = [idxs, dep_input, pos_input, y, lr],  
                        outputs =loss1,
                        updates = updates,
                        allow_input_downcast=True)
                      
pred = theano.function(inputs=[idxs,dep_input,pos_input], 
                       outputs=y_pred, allow_input_downcast=True)       


def prediction(x_set, dep_x_set, pos_x_set, y_set):
    incorrect=0
    N=0
    for i in range(len(x_set)):
        incorrect+=sum(y_set[i]^pred(x_set[i],dep_x_set[i],pos_x_set[i]))
        N+=len(x_set[i])
                
    return (N-incorrect)/float(N)  



nb_epoch = 30
n = 0 
learning_rate=0.006
val_acc1=[]
test_acc1=[]
best=0
import time
while(n<nb_epoch):
    #learning_rate=learning_rate-n*0.0005
    n+=1
    val_acc=[]
    test_acc=[]
    train_data=zip(x_train_set, dep_x_train_set, pos_x_train_set, y_train_set)#
    np.random.shuffle(train_data)
    t0 = time.time()
    for i in range(len(train_data)):
        cost=train(train_data[i][0], train_data[i][1], train_data[i][2], train_data[i][3], learning_rate)
        if i!=0 and i%7999==0:
            print "epoch:", n
            print i,":"
            print "win_size=", win_size
            print "emb_dim=", emb_dim
            print "rnn_hdim=", rnn_hdim
            print "learning_rate:",learning_rate
            print "cost:",cost
            val_value = prediction(x_val_set, dep_x_val_set, pos_x_val_set, y_val_set)
            test_value = prediction(x_test, dep_x_test, pos_x_test, y_test)
            if test_value>best:
                best=test_value
                
            val_acc.append(val_value)
            test_acc.append(test_value)
            print "val_accuracy:", val_value
            print "test_accuracy:", test_value
            print "best till now:",best
            print '\n'
    t1 = time.time()
    print "the hours this epoch takes is:",(t1-t0)/float(3600)
    val_acc1.append(val_acc)
    test_acc1.append(test_acc)
    
