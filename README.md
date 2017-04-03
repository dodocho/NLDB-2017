# NLDB-2017
The datasets this work used are publicly available as follows:<br /> 
GOOGLE dataset(Filippova et al., Sentence compression by  deletion with LSTMs (2015)): http://storage.googleapis.com/sentencecomp/compression-data.json<br /> 
Boardcast dataset(Clarke et al., Constraint-based sentence compression an integer programming approach (2006)): http://jamesclarke.net/research/resources<br /> 
please cite the original papers if you use these datasets.

We employed parser, Parsey McParseface https://github.com/tensorflow/models/tree/master/syntaxnet to yield dependency labels and part-of-speech tags. We used the deep learning framework, Theano(0.8.2), for our experiments.
