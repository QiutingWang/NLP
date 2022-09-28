##Application: 
# machine interpretation: Encoder+Decoder; Seq2Seq
# Text classification with BERT and Graph classification with ViT: Encoder
# Generated model: Decoder

import numpy as np
import torch        #References: https://pytorch.org/docs/stable/torch.html
import torch.nn
import torch.optim as optim #implementing various optimization algorithms,hold the current state and update the parameters by gradients
import matplotlib.pyplot as plt
import math


#pad matrix:  q from encoder, k from decoder
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token 判断哪些是pad过了
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k




#Positional Encoding: reference the formula in paper
class PositionalEncoding(nn.Module):
  def __init__(self, d_model, dropout=0.1, max_len=5000):
    super(PositionalEncoding,self).__init__()
    self.dropout=nn.Dropout(p=dropout)

    pe=torch.zeros(max_len,d_model)
    position=torch.arrange(0,max_len,dtype=torch.float).unqueeze(1)
    div_term=torch.exp(torch.arrange(0, d_model,2).float()*(-math.log(10000)/d_model))
    pe[:,0::2]=torch.sin(position*div_term) #even number
    pe[:,0::1]=torch.cos(position*div_term) #odd number

    pe=pe.unqueeze(0).transpose(0,1)

    self.register_buffer('pe',pe) #the parameter does update
  def forward(self,x):
    """
    x:[seq_len,batch_size,d_model]
    """
    x=x+self.pe[:x.size(0), :]
    return self.dropout(x)




class EncoderLayer(nn.Module):
  def __init__(self):
      super(EncoderLayer, self).__init__()
      self.enc_self_attn = MultiHeadAttention()
      self.pos_ffn = PoswiseFeedForwardNet()

  def forward(self, enc_inputs, enc_self_attn_mask):
      enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V， padding information
      enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
      return enc_outputs, attn




# three parts: Word Embedding, positional embedding, Encoder layer
class Encoder(nn.Model):
  def __init__(self):
    super(Encoder,self).__init__()
    self.src_emb=nn.Embedding(src_vocab_size,d_model) #nn.Embedding:A simple lookup table that stores embeddings of a fixed dictionary and size. Get the word list
    self.pos_emb=PositionalEncoding(d_model)
    # or we use Embedding method: nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_len+1, d_model),freeze=True),we get a position embedding that can update learning
    self.layers=nn.ModuleList([EncoderLayer() for _ in range(n_layers)]) #ModuleList:Holds multi-layers in a list together.
#实现函数:
  def forward(self,enc_inputs):#shape of encoder inputs is [batch size * src_len]
    enc_outputs=self.src_emb(enc_inputs) 
    #output shape:[batchsize, src_len, d_model], use .src_emb() to fix the position with indexing to get a vector
    enc_outputs=self.pos_emb(enc_outputs.transpose(0,1)).transpose(0,1) 
    #position encoding: accept encoder outputs; add the position encoder and word embedding
    enc_self_attn_mask=get_attn_pad_mask(enc_inputs,enc_inputs)
    #tell the model which part is padding

    enc_self_attns=[]
    for layer in self.layers: #把每一层的输出作为下一层的输入
        enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
        enc_self_attns.append(enc_self_attn)
    return enc_outputs, enc_self_attns




# Encoder, Decoder, Output(linear and softmax)
class Transformer(nn.Model): #nn.Model:Base class for all neural network modules.
  def __init__(self):
    super(Transfomer,self).__init__()
    self.encoder=Encoder() #list the 3 main parts in the whole structure
    self.decoder=Decoder()
    self.projection=nn.Linear(d_model,tgt_vocab_size,bias=False) #d_model:dimension of each token decoder layer. list tgt_vocab_size to get softmax shape later,看哪个词出现的probability最大
    #nn.Linear:Applies a linear transformation
  def forward(self,enc_inputs,dec_inputs): #accept two inputs: one from encoder[batch_size,src_len], one from decoder[batch_size,tgt_len]--(shape)

    #For Encoder side:
    enc_outputs,enc_self_attns=self.encoder(enc_inputs) 
    #enc_inputs depends on own function, it can be all tokens, or specific token in each layer, or some parameters
    #enc_self_attns: Q*K matrices production -->softmax transformation, then we get the correlation between this word and other words, for visualization

    #For Decoder side:
    dec_outputs,dec_self_attns,dec_enc_attns=self.decoder(dec_inputs,enc_inputs,enc_outputs)
    #enc_inputs: care about the shape, and which position has been padding
    #dec_self_attns:the correlation between this work and other words, for visualization

    #for output layer:
    dec_logits=self.projection(dec_outputs)  #shape:[batch_size*src_vocab_size*tgt_vocab_size]
    return dec_logits.view(-1,dec_logits.size(-1),enc_self_attns,dec_enc_attns)




if __name__=='__main__':
  #input three sentences:
  sentences=['ich mochte ein bier P','S i want a beer', 'i want a beer E'] #batch size=1
  # S: Symbol that shows starting of decoding input;解码端输入
  # E: Symbol that shows starting of decoding output;解码端真实标签和输出做损失
  # P: Symbol that will fill in blank sequence if current batch data size is short than time steps.（padding)

  #Transformer parameters
  #padding should be 0
  #construct the wordlist, src:source编码端词表, tgt:target解码端词表
  src_vocab={'P':0,'ich':1,'mochte':2,'ein':3,'bier':4}
  src_vocab_size=len(src_vocab) 

  tgt_vocab={'P':0,'i':1,'want':2,'a':3,'beer':4,'S':5,'E':6}
  tgt_vocab_size = len(tgt_vocab)

  src_len = 5 # length of source
  tgt_len = 5 # length of target

  #model parameters setting
  d_model = 512  # Embedding Size
  d_ff = 2048  # FeedForward dimension Linear
  d_k = d_v = 64  # dimension of K(=Q), V
  n_layers = 6  # number of Encoder of Decoder Layer
  n_heads = 8  # number of heads in Multi-Head Attention

  #apply the model
  model = Transformer() #when writing the model, from whole to part; clarify the data-flow shape

