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


# set parameters:
emb_dim = 50
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
rnn_hdim = 150
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
    y_data.append(np.array(temp,dtype=np.int64))
        


embedding = theano.shared(embedding_weights.astype(np.float64))
y = tt.lmatrix('y_label')    
idxs = tt.lmatrix('idxs')
x = embedding[idxs].reshape((idxs.shape[0], emb_dim))


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


emb_dep=np.random.uniform(-0.1, 0.1, (dep_num, dep_dim)).astype(np.float64)
embedding_dep = theano.shared(emb_dep.astype(np.float64))

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



emb_pos=np.random.uniform(-0.1, 0.1, (pos_num, pos_dim)).astype(np.float64)
embedding_pos = theano.shared(emb_pos.astype(np.float64))

pos_input = tt.lmatrix('pos_input')
x_pos = embedding_pos[pos_input].reshape((pos_input.shape[0], pos_dim))

###############################################################################
cat_inputs = tt.concatenate([x, x_dep,x_pos], axis=1)

ini_dim=emb_dim+dep_dim+pos_dim

wx1 = theano.shared(np.random.normal(0, 1/np.sqrt(ini_dim), (ini_dim, rnn_hdim)))
wh1 = theano.shared(np.random.normal(0, 1/np.sqrt(rnn_hdim), (rnn_hdim, rnn_hdim)))
h0_1 = theano.shared(np.zeros(rnn_hdim, ))
bh1  = theano.shared(np.random.normal(0, 1, (rnn_hdim, )))     


wx2 = theano.shared(np.random.normal(0, 1/np.sqrt(ini_dim), (ini_dim, rnn_hdim)))
wh2 = theano.shared(np.random.normal(0, 1/np.sqrt(rnn_hdim), (rnn_hdim, rnn_hdim)))
h0_2 = theano.shared(np.zeros(rnn_hdim, ))
bh2  = theano.shared(np.random.normal(0, 1, (rnn_hdim, )))     


wx3 = theano.shared(np.random.normal(0, 1/np.sqrt(2*rnn_hdim), (2*rnn_hdim, rnn_hdim)))
wh3 = theano.shared(np.random.normal(0, 1/np.sqrt(rnn_hdim), (rnn_hdim, rnn_hdim)))
h0_3 = theano.shared(np.zeros(rnn_hdim, ))
bh3 = theano.shared(np.random.normal(0, 1, (rnn_hdim, )))   

wx4 = theano.shared(np.random.normal(0, 1/np.sqrt(2*rnn_hdim), (2*rnn_hdim, rnn_hdim)))
wh4 = theano.shared(np.random.normal(0, 1/np.sqrt(rnn_hdim), (rnn_hdim, rnn_hdim)))
h0_4 = theano.shared(np.zeros(rnn_hdim, ))
bh4 = theano.shared(np.random.normal(0, 1, (rnn_hdim, )))   

def recurrence1(x_t, h_tm1):
    h_t = tt.tanh(#tt.dot(x_t, wx1) * tt.dot(h_tm1, wh1) + 
                  tt.dot(x_t, wx1) + tt.dot(h_tm1, wh1) + bh1)
    
    return h_t
    
def recurrence2(x_t, h_tm1):
    h_t = tt.tanh(#tt.dot(x_t, wx2) * tt.dot(h_tm1, wh2) + 
                  tt.dot(x_t, wx2) + tt.dot(h_tm1, wh2) + bh2)
    return h_t

def recurrence3(x_t, h_tm1):
    h_t = tt.tanh(#tt.dot(x_t, wx3) * tt.dot(h_tm1, wh3) + 
                  tt.dot(x_t, wx3) + tt.dot(h_tm1, wh3) + bh3)
    return h_t
    
def recurrence4(x_t, h_tm1):
    h_t = tt.tanh(#tt.dot(x_t, wx4) * tt.dot(h_tm1, wh4) + 
                  tt.dot(x_t, wx4) + tt.dot(h_tm1, wh4) + bh4)
    return h_t  
 

    
h1, _ = theano.scan(fn=recurrence1,
                        sequences=cat_inputs,#
                        outputs_info=[h0_1],      
                        n_steps=cat_inputs.shape[0])   
                        
h2, _ = theano.scan(fn=recurrence2,
                        sequences=cat_inputs,#
                        outputs_info=[h0_2],      
                        n_steps=cat_inputs.shape[0],#
                        go_backwards=True)   

hidden12 = tt.concatenate([h1, h2[::-1]], axis=1)


h3, _ = theano.scan(fn=recurrence3,
                        sequences=hidden12,
                        outputs_info=[h0_3],      
                        n_steps=hidden12.shape[0])   
                        
h4, _ = theano.scan(fn=recurrence4,
                        sequences=hidden12,
                        outputs_info=[h0_4],      
                        n_steps=hidden12.shape[0],
                        go_backwards=True)                           

                  

hidden34 = tt.concatenate([h3, h4[::-1]], axis=1)


###############################################################################
gated_dim=emb_dim

wr1 = theano.shared(np.random.normal(0, 1/np.sqrt(2*rnn_hdim+emb_dim), (2*rnn_hdim+emb_dim, gated_dim)))
br1 = theano.shared(np.random.normal(0, 1, (gated_dim, )))    

wr2 = theano.shared(np.random.normal(0, 1/np.sqrt(2*rnn_hdim+dep_dim), (2*rnn_hdim+dep_dim, gated_dim)))
br2 = theano.shared(np.random.normal(0, 1, (gated_dim, ))) 

wr3 = theano.shared(np.random.normal(0, 1/np.sqrt(2*rnn_hdim+pos_dim), (2*rnn_hdim+pos_dim, gated_dim)))
br3 = theano.shared(np.random.normal(0, 1, (gated_dim, ))) 

#wr4 = theano.shared(np.random.normal(0, 1/np.sqrt(2*rnn_hdim+dad_dep_dim), (2*rnn_hdim+dad_dep_dim, gated_dim)))
#br4 = theano.shared(np.random.normal(0, 1, (gated_dim, ))) 

w_cadi = theano.shared(np.random.normal(0, 1/np.sqrt(ini_dim), (ini_dim, gated_dim)))
b_cadi = theano.shared(np.random.normal(0, 1, (gated_dim, ))) 

wz1 = theano.shared(np.random.normal(0, 1/np.sqrt(2*rnn_hdim+emb_dim), (2*rnn_hdim+emb_dim, gated_dim)))
bz1 = theano.shared(np.random.normal(0, 1, (gated_dim, )))    

wz2 = theano.shared(np.random.normal(0, 1/np.sqrt(2*rnn_hdim+dep_dim), (2*rnn_hdim+dep_dim, gated_dim)))
bz2 = theano.shared(np.random.normal(0, 1, (gated_dim, ))) 

wz3 = theano.shared(np.random.normal(0, 1/np.sqrt(2*rnn_hdim+pos_dim), (2*rnn_hdim+pos_dim, gated_dim)))
bz3 = theano.shared(np.random.normal(0, 1, (gated_dim, ))) 

#wz4 = theano.shared(np.random.normal(0, 1/np.sqrt(2*rnn_hdim+dad_dep_dim), (2*rnn_hdim+dad_dep_dim, gated_dim)))
#bz4 = theano.shared(np.random.normal(0, 1, (gated_dim, ))) 

wz5 = theano.shared(np.random.normal(0, 1/np.sqrt(2*rnn_hdim+gated_dim), (2*rnn_hdim+ gated_dim, gated_dim)))
bz5 = theano.shared(np.random.normal(0, 1, (gated_dim, ))) 



def gate_nn(hidden, x, x_dep ,x_pos):#, x_dad_dep
    r1=tt.nnet.sigmoid(tt.dot(tt.concatenate([hidden, x]),     wr1) + br1)
    r2=tt.nnet.sigmoid(tt.dot(tt.concatenate([hidden, x_dep]), wr2) + br2)
    r3=tt.nnet.sigmoid(tt.dot(tt.concatenate([hidden, x_pos]), wr3) + br3)
    #r4=tt.nnet.sigmoid(tt.dot(tt.concatenate([hidden, x_dad_dep]), wr4) + br4)
    w_= tt.tanh(tt.dot(tt.concatenate([x*r1, x_dep*r2, x_pos*r3]), w_cadi) + b_cadi)#, x_dad_dep*r4
    
    z1=tt.exp(tt.dot(tt.concatenate([hidden, x]),     wz1) + bz1)
    z2=tt.exp(tt.dot(tt.concatenate([hidden, x_dep]), wz2) + bz2)
    z3=tt.exp(tt.dot(tt.concatenate([hidden, x_pos]), wz3) + bz3)
    #z4=tt.exp(tt.dot(tt.concatenate([hidden, x_dad_dep]), wz4) + bz4)
    z5=tt.exp(tt.dot(tt.concatenate([hidden, w_]),    wz5) + bz5)
    z_sum=z1+z2+z3+z5#+z4
    
    z1=z1/z_sum
    z2=z2/z_sum
    z3=z3/z_sum
    #z4=z4/z_sum
    z5=z5/z_sum
    
    hz = z1*x + z2*x_dep + z3*x_pos + z5*w_  #z4*x_dad_dep +
    h_cat = tt.concatenate([hz])
    
    
    return h_cat  #z4*x_dad_dep +

    
hidden_gate, _=theano.scan(fn=gate_nn,
                           sequences=[hidden34, x, x_dep ,x_pos],
                           n_steps=x.shape[0])

###############################################################################
cat_dim=gated_dim

w = theano.shared(np.random.normal(0, 1/np.sqrt(cat_dim), 
                                   (cat_dim, rnn_output_dim)))  
b = theano.shared(np.zeros(rnn_output_dim, ))

softmax = tt.nnet.softmax(tt.dot(hidden_gate, w)+b)  


dad_dep_x_train_set = dad_dep_idxs[2000:]
dad_x_train_set = dad_idxs[2000:]
dep_x_train_set = dep_idxs[2000:]
pos_x_train_set = pos_idxs[2000:]
x_train_set = x_data[2000:]
y_train_set = y_data[2000:]


dad_dep_x_val_set = dad_dep_idxs[1000:2000]
dad_x_val_set = dad_idxs[1000:2000]
dep_x_val_set = dep_idxs[1000:2000]
pos_x_val_set = pos_idxs[1000:2000]
x_val_set = x_data[1000:2000]
y_val_set = label[1000:2000]


dad_dep_x_test=dad_dep_idxs[:1000]
dad_x_test=dad_idxs[:1000]
dep_x_test=dep_idxs[:1000]
pos_x_test=pos_idxs[:1000]
x_test=x_data[:1000]
y_test=label[:1000]


params = [embedding, wx1, wh1, h0_1, bh1,
                     wx2, wh2, h0_2, bh2, 
                     wx3, wh3, h0_3, bh3, 
                     wx4, wh4, h0_4, bh4, 
                     
                     wr1, br1,
                     wr2, br2,
                     wr3, br3,
                     #wr4, br4,
                     w_cadi, b_cadi,
                     
                     wz1, bz1,
                     wz2, bz2,
                     wz3, bz3,
                     #wz4, bz4,
                     wz5, bz5,
                     
                     embedding_dep, 
                     embedding_pos, 
                     w, b]

###############################################################################


l2_loss=0
for param in params:
    l2_loss+=(param**2).sum()

lamda=theano.shared(np.array(1e-04))


loss1 = tt.mean(tt.nnet.categorical_crossentropy(softmax, y)) + lamda/float(2) * l2_loss



y_pred = tt.argmax(softmax, axis=1)   

gradients = tt.grad(loss1, params)        

lr=tt.dscalar('lr')
updates = OrderedDict(( p, p-lr*g ) 
                      for p, g in zip( params , gradients))        
 
train = theano.function(inputs  = [idxs, dep_input, pos_input, y, lr],
                        outputs =loss1,
                        updates = updates)
                        #allow_input_downcast=True)
                      
pred = theano.function(inputs=[idxs, dep_input, pos_input], 
                       outputs=y_pred)
                       #allow_input_downcast=True)       


def prediction(x_set,dep_x_set, pos_x_set,y_set):
    incorrect=0
    N=0
    for i in range(len(x_set)):
        incorrect+=sum(y_set[i]^pred(x_set[i],dep_x_set[i],pos_x_set[i]))
        N+=len(x_set[i])
                
    return (N-incorrect)/float(N)  



print 'iteration starts...'
nb_epoch = 30
n = 0
learning_rate=0.007
val_acc1=[]
test_acc1=[]
best=0
import time
while(n<nb_epoch):
    #learning_rate=learning_rate-n*0.0005
    n+=1
    val_acc=[]
    test_acc=[]
    train_data=zip(x_train_set, dep_x_train_set, pos_x_train_set, y_train_set)
    t0 = time.time()
    for i in range(len(train_data)):
        cost=train(train_data[i][0], train_data[i][1], train_data[i][2],train_data[i][3],learning_rate)
        if i!=0 and i%7999==0:
            print "epoch:", n
            print i
            print "emb_dim=", emb_dim
            print "rnn_hdim=", rnn_hdim
            print "learning_rate:",learning_rate
            print "cost:",cost
            val_value = prediction(x_val_set, dep_x_val_set , pos_x_val_set, y_val_set)
            test_value = prediction(x_test,dep_x_test , pos_x_test , y_test)
            if val_value>best:
                best=val_value
                best_test=test_value
            val_acc.append(val_value)
            test_acc.append(test_value)
            print "val_accuracy:", val_value
            print "test_accuracy:", test_value
            print "best till now:",best
            print "best_test under best val:",best_test
            print '\n'
    t1 = time.time()
    print "the hours this epoch takes is:",(t1-t0)/float(3600)
    val_acc1.append(val_acc)
    test_acc1.append(test_acc)
    
