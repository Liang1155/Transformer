import torch
import math
import torch.nn as nn
import torch.nn.functional as F


TEST = False

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout=0.1,max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)


        pe = torch.zeros(max_len,d_model)
        if TEST:
            print(f"pe shape,{pe.shape}")
        position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0)*torch.arange(0,d_model,2).float()/d_model)
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(1)
        if TEST:
            print(f"pe shape2,{pe.shape}")
        self.register_buffer('pe',pe)


    def forward(self,x):
        if TEST:
            print(f"x size,{x.size}")
        x = x + self.pe[:x.size(0),:]
        if TEST:
            print(f"x shape2,{x.shape}")
        return self.dropout(x)


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):

        score = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(self.d_k)
        if TEST:
            print(f"score shape{score.shape}")
        if mask is not None:
            score = score.masked_fill(mask==0,-1e9)

        attn_weights = F.softmax(score,-1)
        output = torch.matmul(attn_weights,V)
        if TEST:
            print(f"output shape,{output.shape}")
        return output,attn_weights

    def split_heads(self, x):

        batch_size,seq_len,_ = x.size()
        x = x.view(batch_size,seq_len,self.num_heads,self.d_k)
        return x.transpose(2,1)

    def combine_heads(self, x):

        batch_size,_,seq_len,_ = x.size()
        x = x.transpose(1,2).contiguous()
        x = x.view(batch_size,seq_len,self.d_model)
        return x


    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)

        output = self.W_o(self.combine_heads(attn_output))
        return output


class FeedForward(nn.Module):

    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model,d_ff)
        self.fc2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DecoderLayer(nn.Module):

    def __init__(self,d_model,num_heads,d_ff,dropout=0.1,):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model,num_heads)
        self.ffn = FeedForward(d_model,d_ff,dropout)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, tgt_mask=None):
        attn1 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))

        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x,tgt_mask=None):
        for layer in self.layers:
            x = layer(x, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):

    def __init__(
        self,
        tgt_vocab_size,   # 目标语言词表大小
        d_model=512,      # 模型维度（论文默认512）
        num_heads=8,      # 注意力头数（论文默认8）
        num_layers=6,     # Encoder/Decoder 层数（论文默认6）
        d_ff=2048,        # FFN 中间层维度（论文默认2048）
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()

        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)

        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)

        self.output_linear = nn.Linear(d_model, tgt_vocab_size)

        self.d_model = d_model
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_tgt_mask(self, tgt, pad_idx=0):

        tgt_len = tgt.size(1)
        pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, tgt_len]

        # 下三角矩阵：位置 i 只能看到 0..i
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)    # [1, 1, tgt_len, tgt_len]

        return pad_mask & causal_mask

    def forward(self, tgt, pad_idx=0):

        tgt_mask = self.make_tgt_mask(tgt, pad_idx)

        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = tgt_emb.transpose(0, 1)
        tgt_emb = self.pos_encoding(tgt_emb).transpose(0, 1)
              # [batch, src_len, d_model]

        # 解码
        dec_output = self.decoder(tgt_emb, tgt_mask)  # [batch, tgt_len, d_model]

        # 输出层：每个位置预测下一个词的概率
        logits = self.output_linear(dec_output)                        # [batch, tgt_len, tgt_vocab_size]
        return logits



if __name__ == "__main__":
    with open("Shakespeare.txt" ,"r",encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # 超参数
    tgt_vocab_size = vocab_size
    batch_size     = 32
    src_len        = 10
    tgt_len        = 8
    device = torch.device("cuda")

    model = Transformer(
        tgt_vocab_size=tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        dropout=0.1
    )
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))


    stoi = {ch:i for i ,ch in enumerate(chars)}
    itos = {i:ch for i ,ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(l):
        return [itos[c] for c in l]

    data = torch.tensor(encode(text),dtype=torch.long)

    def get_batch(data,batch_size,block_size):
        ix = torch.randint(0,len(data)-block_size,(batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])

        return x,y


    model = model.to(device)
    model.load_state_dict(
        torch.load("shakespear_gpt_weights.pth",map_location=device)
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr = 3e-4)

    batch_size = 32
    block_size = 256
    def generate_text(model,start_str,max_new_tokens,block_size,device):
        model.eval()
        context = torch.tensor(encode(start_str),dtype=torch.long,device=device).unsqueeze(0)
        for _ in range(max_new_tokens):
            cond_context = context[:,-block_size:]

            with torch.no_grad():
                logits = model(cond_context)

                logits = logits[:,-1,:]
                probs = F.softmax(logits,dim=-1)
                next_token = torch.multinomial(probs,num_samples=1)
                context = torch.cat((context,next_token),dim=1)

        return ''.join(decode(context[0].tolist()))

    print(generate_text(model,start_str=") Romeo,",max_new_tokens=500,block_size=8,device=device))
