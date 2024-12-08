import torch
import torch.nn as nn
from models.transformer.attention import MultiHeadAttention


class MultiModalAttention(nn.Module):
    def __init__(self, depth=1, hidden_size=128, head=4):
        super(MultiModalAttention, self).__init__()
        self.depth = depth
        self.hidden_size = hidden_size
        # input token _ encode_state
        self.lang_box_attn = nn.ModuleList(MultiHeadAttention(
            d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head)for i in range(self.depth))
        self.box_lang_attn = nn.ModuleList(MultiHeadAttention(
            d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head)for i in range(self.depth))
        self.lang_pc_attn = nn.ModuleList(MultiHeadAttention(
            d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head)for i in range(self.depth))
        self.box_pc_attn = nn.ModuleList(MultiHeadAttention(
            d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head)for i in range(self.depth))

    def forward(self, lang_emb, box_emb, pc_emb):
        lang_out = lang_emb  # (B, 50, 128)
        box_out = box_emb  # (B, num_proposal, 128)
        pc_out = pc_emb  # (B, voxel_size, 128)

        for i in range(self.depth):
            lang_out = self.lang_pc_attn[i](lang_out, pc_out, pc_out)
            box_out = self.box_pc_attn[i](box_out, pc_out, pc_out)
            lang_out2 = self.lang_box_attn[i](lang_out, box_out, box_out)
            box_out2 = self.box_lang_attn[i](box_out, lang_out, lang_out)
            lang_out = lang_out2
            box_out = box_out2
        return lang_out, box_out


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class CrossAttentionDecoderLayer(nn.Module):

    def __init__(self, ffn_hidden=256, hidden_size=128, head=4, drop_prob=.1):
        super(CrossAttentionDecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head)

        self.enc_dec_attention = MultiHeadAttention(
            d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head)
        self.ffn = PositionwiseFeedForward(
            d_model=hidden_size, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=drop_prob)
        self.head = head

    def forward(self, query, key, value, src_mask=None, src_trg_mask=None):
        # 1. compute self attention
        # 2. add and norm
        _x = query
        x = self.self_attention(query, query, query, attention_mask=src_mask)

        # 3. compute encoder - decoder attention
        # 4. add and norm
        _x = x
        x = self.enc_dec_attention(x, key, value, attention_mask=src_trg_mask)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 6. add and norm
        x = self.dropout(x)
        x = self.norm(x + _x)
        return x
