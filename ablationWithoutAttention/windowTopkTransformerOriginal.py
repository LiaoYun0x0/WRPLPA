import torch
import torch.nn as nn
from typing import Callable, Optional, Union, Tuple, List

def qk_map(x):
    return torch.nn.functional.elu(x) + 1

class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.qk_map = qk_map
        self.eps = eps

    def forward(self, q, k, v, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, V]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, V)
        """
        Q = self.qk_map(q)
        K = self.qk_map(k)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            v = v * kv_mask[:, :, None, None]

        v_length = v.size(1)
        v = v / v_length  # prevent fp16 overflow

        # KV = torch.matmul(K.permute(0,2,3,1), V.permute(0,2,1,3))
        KV = torch.einsum("nshd,nshv->nhdv", K, v)  # (S,D)' @ S,V

        # compuet  similarity between each query and the mean of keys  
        # Z = 1 / ( (Q * (K.sum(dim=1)+self.eps)).sum(-1) ) 
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)


        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()

class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(EncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class FullAttention(nn.Module):
    def __init__(self):
        super(FullAttention, self).__init__()

    def forward(self, query, key, value): 
        #         # Compute the unnormalized attention and apply the masks
        # QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        # if kv_mask is not None:
        #     QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

        # # Compute the attention and the weighted average
        # softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        # A = torch.softmax(softmax_temp * QK, dim=2)
        # if self.use_dropout:
        #     A = self.dropout(A)

        # queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        # return queried_values.contiguous()
        # 计算注意力分数
        b,num_heads,n,head_dim = query.shape
        q = query.transpose(1, 2).view(b, -1, num_heads * head_dim) #[100,64,256]
        k = key.transpose(1, 2).view(b, -1, num_heads * head_dim) #[100,64,256]
        # 归一化注意力分数
        attention_scores = torch.softmax(torch.matmul(q,k.transpose(1,2)),dim=-1)

        A = torch.matmul(query, key.transpose(-2, -1))  #query:[100,8,64,32] [b*num_windows,num_heads,n,head_dim]
        softmax_temp = 1. / query.size(3)**.5
        attention_weights = torch.softmax(A * softmax_temp, dim=-1)  #[100,8,64,64]
        
        # 使用注意力权重加权平均值来计算输出
        output = torch.matmul(attention_weights, value)
        
        return output, attention_scores

class MultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.attention = FullAttention()
        self.merge = nn.Linear(embed_dim, embed_dim, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim*2, bias=False),
            nn.ReLU(True),
            nn.Linear(embed_dim*2, embed_dim, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x,source): #x:[b*num_window,c,window_size,window_size]
        b,c,h,w = x.shape #b: batch_size * nums_window
        x = x.view(b,c,h*w).permute(0,2,1) #[b*num_window,c,window_size**2]
        source = source.view(b,c,h*w).permute(0,2,1)
        b, n, embed_dim = x.size()
        # 将输入分割成多个头
        query = self.query(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)#[b*num_window,num_heads,window_size**2,head_dim]
        key = self.key(source).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(source).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 使用LinearAttention计算自注意力
        attention_output, attn_score = self.attention(query, key, value) 

        attention_output = self.merge(attention_output.view(b, -1, self.num_heads*self.head_dim))  # [N, L, C]
        attention_output = self.norm1(attention_output)

        # feed-forward network
        attention_output = self.mlp(torch.cat([x, attention_output], dim=2))
        attention_output = self.norm2(attention_output) 
        
        # 将多个头的输出连接起来
        # attention_output = attention_output.transpose(1, 2).contiguous().view(b, n, embed_dim)
        
        return x + attention_output,attn_score


def window_partition_nchw(x, window_size: List[int]):
    B, C, H, W = x.shape
    assert(H % window_size[0] == 0, f'height ({H}) must be divisible by window ({window_size[0]})')
    assert(W % window_size[1] == 0, '')
    x = x.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size[0], window_size[1])
    return windows

def window_reverse_nchw(windows, window_size: List[int], img_size: List[int]):#[b*num_window,window_size*window_size,c]
    H, W = img_size
    C = windows.shape[-1]
    windows = windows.permute(0,2,1).view(-1,C,window_size[0],window_size[1])
    x = windows.view(-1, H // window_size[0], W // window_size[1], C, window_size[0], window_size[1])
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, C, H, W)
    return x

class WindowTopKAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, topk):
        super(WindowTopKAttention, self).__init__()
        self.window_attention = MultiheadSelfAttention(embed_dim, num_heads)
        self.window_size = window_size
        self.topk = topk
        self.linear_attention = EncoderLayer(embed_dim,num_heads)
        self.dim = embed_dim

    def forward(self, x,source,mask0=None,mask1=None): #x: [b,c,h,w]
        x0 = x
        b,embed_dim,h,w = x.shape#[4,256,40,40]
        n = h*w
        nums_window = (h//self.window_size) * (w//self.window_size) #[25]
        #b, n, embed_dim = x.size()
        # 将输入划分成窗口
        x = window_partition_nchw(x,[self.window_size,self.window_size]) #x:[b*num_window,c,window_size,window_size] [100,256,8,8]
        source = window_partition_nchw(source,[self.window_size,self.window_size])
        # 计算每个窗口的注意力
        '''bxnxhxwx1'''
        """
            根据attn_score选择 window_attntion的topk
            注意力的q * k的结果
            # 计算注意力分数矩阵（注意力分数矩阵的形状为[batch_size, num_queries, num_keys]）
            attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # 注意力分数计算

            # 选择前k个最大的分数对应的索引
            top_k_scores, top_k_indices = torch.topk(attention_scores, k=top_k, dim=-1)

            # 从值矩阵中提取相应的值，构建top-k注意力输出
            top_k_values = torch.gather(V, dim=-1, index=top_k_indices.unsqueeze(-1).expand(-1, -1, -1, -1, top_k))
        """
        window_output, attn_score = self.window_attention(x,source) #[b*num_window,h*w,c]   attn_score:[100,64,64]
        
        attn_score = torch.softmax(attn_score, dim=-1) * torch.softmax(attn_score, dim=-2)
        topk_indices = torch.topk(attn_score, self.topk, dim=1, largest=True).indices
        window_attention_topk = torch.gather(window_output, dim=2, index=topk_indices)
        # 根据attn_score,选择每个窗口的topk个patches;  attn_score:[]
        # window_attention_topk, indices = window_output.topk(self.topk, dim=-2) # [b*num_window,topk,c]
        
        window_attention_topk = window_attention_topk.view(b,nums_window*self.topk,embed_dim)
        
        # 使用LinearAttention计算窗口之间的注意力
        window_attention_output = self.linear_attention(window_attention_topk, window_attention_topk) #[b*num_window,topk,c]
        window_attention_output = window_attention_output.view(b*nums_window,self.topk,embed_dim)

        # 构建一个和 window_attention_output 一样的 mask 张量
        mask = torch.zeros_like(window_attention)

        # 使用索引操作将 window_attention_output 的值加到 mask 中的对应位置
        mask.scatter_(1, indices, window_attention_output)

        # 将 mask 的值加到 window_attention 中
        window_attention += mask  #[256,25,256]

        win_output = window_reverse_nchw(window_attention,[self.window_size,self.window_size],[h,w])

        '''
            # 将融合后的结果与原始输入进行连接或叠加
                fused_output = torch.cat([x, win_output], dim=-1)  # 连接方式
                
                # 对融合后的表示进行降维
                fused_output = self.reduce_dimension(fused_output)
        '''   
        #或者叠加
        '''不融合试试 我记得没融合吧'''
        #fused_output = x0 + win_output  # 使用恒等映射来保持维度相同
        return win_output #[b,c,h,w]

    def reduce_dimension(self, x):
        return nn.Linear(x.size(-1), self.dim)(x)


