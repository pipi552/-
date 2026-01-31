
from torch.nn.functional import relu
from .st_gcn import *

class UltraPoserNetwork(nn.Module):

    def __init__(self, n_input, n_output, n_feature_embed=256, n_joint_embed=512, n_layer_rnn=2, n_layer_trans=2, n_head=8, dropout=0.2):
        super(UltraPoserNetwork, self).__init__()

        self.imu_len = n_input["imu"]
        self.doppler_len = n_input["doppler"]
        self.cir_len = n_input["cir"]
        self.n_joints = 24
        self.n_feature_embed = n_feature_embed
        self.imu_encoder = IMUEncoder(n_input=self.imu_len, n_embed=n_feature_embed//2, n_layer=n_layer_rnn, bidirectional=True)
        self.doppler_encoder = UltrasoundEncoder(n_input=self.doppler_len, n_embed=n_feature_embed, n_layer=n_layer_trans, n_head=n_head, dropout=dropout)

        self.device_doppler_encoder = UltrasoundEncoder(n_input=self.doppler_len//3*2, n_embed=n_feature_embed,
                                                       n_layer=n_layer_trans, n_head=n_head, dropout=dropout)
        self.cir_encoder = UltrasoundEncoder(n_input=self.cir_len, n_embed=n_feature_embed, n_layer=n_layer_trans,
                                             n_head=n_head, dropout=dropout)
        self.imu_cross_attn_block = CrossAttentionBlock(n_feature_embed, num_heads=8)
        self.ultra_cross_attn_block = CrossAttentionBlock(n_feature_embed, num_heads=8)

        graph_cfg = {
            'layout': 'ultrapose',
            'strategy': 'spatial',
            'max_hop': 1,
            'dilation': 1
        }
        self.stgcn = ST_GCN_18(in_channels=n_feature_embed, num_class=10, graph_cfg=graph_cfg, dropout=dropout, edge_importance_weighting=True)

        self.global_joint_rotation_decoder = nn.Sequential(
            nn.Linear(n_feature_embed, n_joint_embed),
            nn.ReLU(),
            nn.Linear(n_joint_embed, n_feature_embed * self.n_joints)
        )
        self.motion_decoder = nn.Sequential(
            nn.Linear(32 * self.n_joints, n_feature_embed),
            nn.ReLU(),
            nn.Linear(n_feature_embed, n_output)
        )

        self.right_body_index = torch.tensor([23, 21, 19, 17, 2, 5, 8, 11])
        self.left_body_index = torch.tensor([22, 20, 18, 16, 1, 4, 7, 10])
        self.imu_body_index_hand = torch.tensor([15, 18, 19])
        self.imu_body_index_pocket = torch.tensor([15, 18, 2])
        self.n_left_joints = len(self.left_body_index)
        self.n_right_joints =len(self.right_body_index)
        self.n_imu_joints = len(self.imu_body_index_hand)

        self.right_joint_rotation_decoder = nn.Sequential(
            nn.Linear(n_feature_embed, n_joint_embed),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_joint_embed, n_feature_embed * self.n_right_joints)
        )

        self.left_joint_rotation_decoder = nn.Sequential(
            nn.Linear(n_feature_embed, n_joint_embed),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_joint_embed, n_feature_embed * self.n_left_joints)
        )

        self.imu_joint_rotation_decoder = nn.Sequential(
            nn.Linear(n_feature_embed, n_joint_embed),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_joint_embed, n_feature_embed * self.n_imu_joints)
        )

    def forward(self, x, x_lens=None, h=None):
        imu_feature = self.imu_encoder(x[:, :, 0:self.imu_len], x_lens)
        ultrasound_feature = self.doppler_encoder(x[:, :, self.imu_len:self.imu_len + self.doppler_len])

        cir_feature = self.cir_encoder(x[:, :, self.imu_len + self.doppler_len:-1])

        enhanced_imu = self.imu_cross_attn_block(imu_feature, ultrasound_feature * cir_feature)
        enhanced_ultra = self.ultra_cross_attn_block(ultrasound_feature * cir_feature, imu_feature)

        global_joint_rotation = self.global_joint_rotation_decoder(enhanced_imu * enhanced_ultra)
        N, T, _ = global_joint_rotation.size()
        global_joint_rotation = global_joint_rotation.view(N, T, self.n_joints, -1)

        doppler_phone = torch.cat(
            [x[:, :, self.imu_len:self.imu_len + 100], x[:, :, self.imu_len + 200:self.imu_len + 300]], dim=2)
        doppler_watch = torch.cat(
            [x[:, :, self.imu_len + 100:self.imu_len + 200], x[:, :, self.imu_len + 200:self.imu_len + 300]], dim=2)

        phone_doppler_feature = self.device_doppler_encoder(doppler_phone)
        watch_doppler_feature = self.device_doppler_encoder(doppler_watch)

        right_joint_rotation = self.right_joint_rotation_decoder(phone_doppler_feature * cir_feature).view(N, T, self.n_right_joints, -1)
        left_joint_rotation = self.left_joint_rotation_decoder(watch_doppler_feature).view(N, T, self.n_left_joints, -1)

        imu_joint_rotation = self.imu_joint_rotation_decoder(imu_feature).view(N, T, self.n_imu_joints, -1)
        hand_idx = torch.unique(torch.where(x[:, :, -1] == 0)[0])
        pocket_idx = torch.unique(torch.where(x[:, :, -1] == 1)[0])
        hand_size = len(hand_idx)
        pocket_size = len(pocket_idx)
        global_joint_rotation[:, :, self.right_body_index] *= right_joint_rotation
        global_joint_rotation[:, :, self.left_body_index] *= left_joint_rotation
        if hand_size !=0:
            global_joint_rotation[hand_idx][:, :, self.imu_body_index_hand] *= imu_joint_rotation[hand_idx]
        if pocket_size !=0:
            global_joint_rotation[pocket_idx][:, :, self.imu_body_index_pocket] *= imu_joint_rotation[pocket_idx]

        global_joint_rotation = global_joint_rotation.view(N, T, self.n_joints, self.n_feature_embed, 1).permute(0, 3, 1, 2, 4).contiguous()
        _, feature = self.stgcn.extract_feature(global_joint_rotation)
        feature = feature.permute(0, 2, 3, 1, 4).contiguous()
        feature = feature.view(N, 30, -1)
        output_joint_rotation = self.motion_decoder(feature)
        return output_joint_rotation


class UltrasoundEncoder(nn.Module):

    def __init__(self, n_input, n_embed=256, n_layer=2, n_head=8, dropout=0.2):
        super(UltrasoundEncoder, self).__init__()

        self.linear_embedding = nn.Linear(n_input, n_embed)
        encoder_layer = nn.TransformerEncoderLayer(n_embed, nhead=n_head)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.linear_embedding(self.dropout(x))
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)

        return x


class IMUEncoder(nn.Module):

    def __init__(self, n_input, n_embed, n_layer=2, bidirectional=True, dropout=0.2):
        super(IMUEncoder, self).__init__()
        self.rnn = nn.LSTM(n_embed, n_embed, n_layer, bidirectional=bidirectional, batch_first=True)
        self.linear = nn.Linear(n_input, n_embed)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_lens=None, h=None):

        x = relu(self.linear(self.dropout(x)))
        x = nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        x, h = self.rnn(x, h)
        x, output_lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        return x



class CrossAttention(nn.Module):
    def __init__(self, n_embed, num_heads):

        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=n_embed,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x_a, x_b):

        query = x_a
        key = x_b
        value = x_b

        attn_output, _ = self.multihead_attn(query=query, key=key, value=value)
        return attn_output


class CrossAttentionBlock(nn.Module):

    def __init__(self, n_embed, num_heads):
        super().__init__()
        self.attention = CrossAttention(n_embed, num_heads)
        self.norm = nn.LayerNorm(n_embed)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_a, x_b):

        residual = x_a
        attn_output = self.attention(x_a, x_b)
        output = self.relu(residual + self.dropout(attn_output))
        return output