import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 32 #how many independent samples we have
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32

torch.manual_seed(1337)

with open('shakespeare.txt','r',encoding='utf-8') as f:
    text = f.read()

#here are all the unique characters that occur in this text
chars = sorted(list(set(text))) 
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)} #enumerate - allows you to iterate over a sequence while also keeping track of the index
itos = {i:ch for i,ch in enumerate(chars)} #creates a dictionary with the index as the key and the character as the value
encode = lambda x: [stoi[ch] for ch in x]
decode = lambda x: ''.join([itos[i] for i in x])

data = torch.tensor(encode(text),dtype=torch.long)
n = int(0.9*len(data)) #first 90% of the data
train_data, val_data = data[:n], data[n:] #splitting the data into training and validation sets

#data loading
def get_batch(split):
    #generate a samll batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device),y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits,loss= model(X,Y)
            losses[k] = loss.item()
        out[split]=losses.mean()
    model.train()
    return out

class Head(nn.Module):
    "one head of slef attention"

    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd,head_size,bias=False)
        self.query = nn.Linear(n_embd,head_size,bias=False)
        self.value = nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
    
    def forward(self,x):
        B,T,C = x.shape
        key = self.key(x)
        query = self.query(x)
        #compute attention scores
        wei = query @ key.transpose(-2,-1) * C**-0.5 # B,T,C @ B,C,T -> B,T,T scaled attenstion to reduce variance
        wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf')) #lower triangular mask
        wei = F.softmax(wei,dim=-1)
        #perform he weighted aggreagation values
        v = self.value(x) # B,T,C
        out = wei @ v # B,T,T @ B,T,C -> B,T,C
        return out
    
class MultiHeadAttention(nn.Module):
    "multi-head attention"

    def __init__(self,num_heads,head_size):
        super().__init__()
        self._heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd,n_embd)

    def forward(self,x):
        out = torch.cat([h(x) for h in self._heads],dim=-1)
        out = self.proj(out)
        return out
    
class FeedForward(nn.Module):
    "simple linear layer followed by a non linearity"
    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd)
        )
    def forward(self,x):
        return self.net(x)
    
class Block(nn.Module):
    "communication followed by computation"
    def __init__(self,n_embd,n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head,head_size)
        self.ff = FeedForward(n_embd)
    
    def forward(self,x):
        x = x + self.sa(x)
        x = x + self.ff(x)
        return x
    


#super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        #each token directly reads off the logits from the embedding layer
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)   
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd,4),
            Block(n_embd,4),
            Block(n_embd,4),
        )
        #self.sa_head = Head(n_embd)
        #self.sa_heads = MultiHeadAttention(4, n_embd//4)
        #self.ff = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)

    def forward(self,idx,targets=None):
        #idx and targets are both tensor or integers of shape (batch_size,block_size)
        #we want to predict the next token for each token in the block
        #so we need to compute the logits for each token in the block
        #we can do this by using the token_embedding_table
        #the token_embedding_table is a square matrix of shape (vocab_size,vocab_size)
        B,T = idx.shape
        token_embed = self.token_embedding_table(idx) #token_embed is (B,T,C)
        pos_embed = self.position_embedding_table(torch.arange(T,device=device)) #pos_embed is (T,C)
        x = token_embed + pos_embed
        x = self.sa_heads(x) # apply one head of self attention
        #x = self.ff(x) # apply feed forward
        x = self.blocks(x)
        logits = self.lm_head(x) # logits is (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            #we want to predict the next token for each token in the block
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits,loss
    
    def generate(self,idx,max_new_tokens):
        #idx is (B,T) array of indices in the current context
        #max_new_tokens is the maximum number of tokens to generate
        for _ in range(max_new_tokens):
            #crop  idx to the last block_size tokens
            idx_cond = idx[:,-block_size:] #idx_cond is (B,block_size)
            #get the predictions for the next token
            logits,loss = self(idx_cond)
            #focus only on the last time step
            logits = logits[:,-1,:] #becomes (B,C)
            #apply softmax to get probabilities
            probs = F.softmax(logits,dim=-1)
            #sample from the distribution
            idx_next = torch.multinomial(probs,num_samples=1) #idx_next is (B,1)
            #append the sampled index to the running sequence
            idx = torch.cat((idx,idx_next),dim=1) #idx is (B,T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

#create a pytorch optimizer
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

for iter in range(max_iters):

    #every once in a while evaluate the loss on the training and validation sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'step {iter}: train loss {losses["train"]:.2f}, val loss {losses["val"]:.2f}')
    
    #sample a batch of data
    xb,yb = get_batch('train')

    #evaluate the loss 
    logits,loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate some text
context = torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(m.generate(context,max_new_tokens=500)[0].tolist()))
