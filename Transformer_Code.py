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


def make_batch(sentences):
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)
    # use LongTensor to change format



#Two Padding Masks: Self-Attention Layer, Interactive Attention layer
#Self-attention:pad matrix:  q from encoder, k from decoder
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size() 
    batch_size, len_k = seq_k.size()
    # eq(zero) is padding token, computes element-wise equality
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)    # batch_size x 1 x len_k(=len_q), one is masking 
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

#Sequential mask:
def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask


#ScaledDotProductAttention
class ScaledDotProductAttention(nn.Module):
  def __init__(self): #创建实例的时候 put in 一些属性
    super(ScaledDotProductAttention,self).__init__() 
    #super():Create a class that will inherit all the methods and properties from another class

  def forward(self, Q, K, V, attn_mask):
    scores=torch.matmul(Q,K.transpose(-1,-2))/np.sqrt(d_k)
    scores.masked_fill_(attn_mask, -1e9) #fill the mask with infinte negative number
    attn = nn.Softmax(dim=-1)(scores) #after softmax process, the value in position mask in 0
    context = torch.matmul(attn, V)
    return context, attn


#Multihead Attention
class MultiHeadAttention(nn.Module):
    def __init__(self): 
        super(MultiHeadAttention, self).__init__()
        #input the same Q/K/V, we use linear to get mapping parameter matrices Wq, Wk, Wv. Then we get the same dimension
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model) #layer normalization

    def forward(self, Q, K, V, attn_mask):
        #Shape: Q: [batch_size x len_q x d_model], K: [batch_size x len_k x d_model], V: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split with view function, dimension of head is d_k-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]. #torch.unsqueeze:Returns a new tensor with a dimension of size one inserted at the specified position. #give each head some information let them calculate

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask) #achieve the formula of Attention(Q,K,v) in paper
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]


#PowiseFeedForwardNet(前馈神经网络)
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2))) #torch.transpose(input, dim0, dim1)
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)



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



#EncoderLayer: MultiHeadAttention & PoswiseFeedForwardNet
class EncoderLayer(nn.Module):
  def __init__(self):
      super(EncoderLayer, self).__init__()
      self.enc_self_attn = MultiHeadAttention()
      self.pos_ffn = PoswiseFeedForwardNet()

  def forward(self, enc_inputs, enc_self_attn_mask):
     #the following is self-attention layer, input:enc_inputs in three times
      enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V， padding information
      enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
      return enc_outputs, attn



# DecoderLayer:
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn




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




#Decoder part:
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs): # dec_inputs : [batch_size x target_len]
        dec_outputs = self.tgt_emb(dec_inputs) #[batch_size,tgt_len,d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0,1)).transpose(0,1) #[batch_size,tgt_len,d_model]

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs) #self-attention part padding position

        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs) #we put 1 in the right-up corner in the matrix square, masking the latter information in self-attention layer

        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0) # add up two matrices, if addition>0-->1; if addition<=0-->0;

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs) 
        #interactive attention layer

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns




#Transformer Model: Encoder, Decoder, Output(linear and softmax)
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





#Visualization:
def showgraph(attn):
    attn = attn[-1].squeeze(0)[0]
    attn = attn.squeeze(0).data.numpy()
    fig = plt.figure(figsize=(n_heads, n_heads)) # [n_heads, n_heads]
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attn, cmap='viridis')
    ax.set_xticklabels(['']+sentences[0].split(), fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels(['']+sentences[2].split(), fontdict={'fontsize': 14})
    plt.show()



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
  model = Transformer() #when writing the model, from whole to part; clarify the data-flow shape....

  #compute loss function and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
