# =============================================================================
# Author: Oscar Gavin, ogavin1@shef.ac.uk
# Edited version of MONAI 3D UNETR model: https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV
# =============================================================================


import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SingleDeconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)


class Conv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv2DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv2DBlock(in_planes, out_planes),
            SingleConv2DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class SelfAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(embed_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(embed_dim, self.all_head_size)
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)

        self.out = nn.Linear(self.all_head_size, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

        self.vis = False

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(
            hidden_states.size(0), -1, hidden_states.size(1))
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, in_features, act_layer=nn.ReLU, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=768, d_ff=2048, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.w_2(x)
        return x


class Embeddings(nn.Module):
    def __init__(self, input_dim, embed_dim, image_size, patch_size, dropout):
        super().__init__()
        self.n_patches = int(
            (image_size[0] * image_size[1]) / (patch_size * patch_size))
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv2d(in_channels=input_dim, out_channels=embed_dim,
                                          kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.n_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1, image_size=(512, 512), patch_size=16):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_dim = int(
            (image_size[0] * image_size[1]) / (patch_size * patch_size))
        self.mlp = PositionwiseFeedForward(embed_dim, 2048)
        self.attn = SelfAttention(
            num_heads, embed_dim, dropout)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)

        # Add this code to check the size of the input tensor to the SelfAttention module
        assert x.size() == h.size(
        ), f"Shape mismatch: {x.size()} != {h.size()}"

        x = x + h
        h = x

        x = self.mlp_norm(x)
        x = self.mlp(x)

        x = x + h
        return x, weights


class Transformer(nn.Module):
    def __init__(self, input_dim, embed_dim, image_size, patch_size, num_heads, num_layers, dropout, extract_layers):
        super().__init__()
        self.embeddings = Embeddings(input_dim=input_dim, embed_dim=embed_dim,
                                     image_size=image_size, patch_size=patch_size, dropout=dropout)
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.extract_layers = extract_layers
        for _ in range(num_layers):
            layer = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads,
                                     dropout=dropout, image_size=image_size, patch_size=patch_size)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        extract_layers = []
        hidden_states = self.embeddings(x)

        for depth, layer_block in enumerate(self.layer):
            hidden_states, _ = layer_block(hidden_states)
            if depth + 1 in self.extract_layers:
                extract_layers.append(hidden_states)

        return extract_layers


class UNETR(nn.Module):
    def __init__(self, image_size=(512, 512), input_dim=1, output_dim=1, embed_dim=768, patch_size=16, num_heads=12, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = 12
        self.ext_layers = [3, 6, 9, 12]

        self.patch_dim = [int(x / patch_size) for x in image_size]

        # Transformer Encoder
        self.transformer = \
            Transformer(
                input_dim,
                embed_dim,
                image_size,
                patch_size,
                num_heads,
                self.num_layers,
                dropout,
                self.ext_layers
            )

        # U-Net Decoder
        self.decoder0 = \
            nn.Sequential(
                Conv2DBlock(input_dim, 32, 3),
                Conv2DBlock(32, 64, 3)
            )

        self.decoder3 = \
            nn.Sequential(
                Deconv2DBlock(embed_dim, 512),
                Deconv2DBlock(512, 256),
                Deconv2DBlock(256, 128)
            )

        self.decoder6 = \
            nn.Sequential(
                Deconv2DBlock(embed_dim, 512),
                Deconv2DBlock(512, 256),
            )

        self.decoder9 = \
            Deconv2DBlock(embed_dim, 512)

        self.decoder12_upsampler = \
            SingleDeconv2DBlock(embed_dim, 512)

        self.decoder9_upsampler = \
            nn.Sequential(
                Conv2DBlock(1024, 512),
                Conv2DBlock(512, 512),
                Conv2DBlock(512, 512),
                SingleDeconv2DBlock(512, 256)
            )

        self.decoder6_upsampler = \
            nn.Sequential(
                Conv2DBlock(512, 256),
                Conv2DBlock(256, 256),
                SingleDeconv2DBlock(256, 128)
            )

        self.decoder3_upsampler = \
            nn.Sequential(
                Conv2DBlock(256, 128),
                Conv2DBlock(128, 128),
                SingleDeconv2DBlock(128, 64)
            )

        self.decoder0_header = \
            nn.Sequential(
                Conv2DBlock(128, 64),
                Conv2DBlock(64, 64),
                SingleConv2DBlock(64, output_dim, 1)
                # potential change for landmark localisation rather than image segmentation
                # SingleConv2DBlock(64, 2, 1),  # change output dimension to 2 for heatmap
                # nn.Softmax2d()  # apply softmax to produce heatmap
            )

    def forward(self, x):
        z = self.transformer(x)
        z0, z3, z6, z9, z12 = x, *z
        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)

        # ===================================
        # Produce Co-ordinates
        # ===================================
        # Append the coordinates to the features map
        # coords = coords.unsqueeze(-1).unsqueeze(-1)
        # coords = coords.repeat(1, 1, x.shape[2], x.shape[3])
        # z0 = torch.cat([z0, coords], dim=1)

        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))

        if target.shape[-2:] != input.shape[-2:]:
            target = F.interpolate(
                target, size=input.shape[-2:], mode='bilinear', align_corners=False)
        return F.mse_loss(input, target, reduction=self.reduction)
