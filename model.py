import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from util import *

'''***********************************************************************'''
''' Loss/criterion, and metrics '''
'''***********************************************************************'''


# rmse: root of mean square error
class MyRmseMetric(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, y_pred, y_true):
        return torch.square(F.mse_loss(y_pred, y_true))


def compute_mape(y_pred, y_true, eps=1e-6):
    assert y_pred.shape == y_true.shape

    relative_err = torch.abs((y_true - y_pred) / y_true)
    relative_err[y_true.abs() < eps] = torch.nan
    return torch.nanmean(relative_err)


# mape: mean absolute percentage error
class MyMapeMetric(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps = config.computation_eps

    def forward(self, y_pred, y_true):
        return compute_mape(y_pred, y_true, self.eps)


# mapa: mean absolute percentage accuracy
class MyMapaMetric(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps = config.computation_eps

    def forward(self, y_pred, y_true):
        return (1 - compute_mape(y_pred, y_true, self.eps))


def compute_mae(y_pred, y_true):
    assert y_pred.shape == y_true.shape

    absolute_err = torch.abs(y_true - y_pred)
    return torch.nanmean(absolute_err)


# mape: mean absolute error
class MyMaeMetric(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, y_pred, y_true):
        return compute_mae(y_pred, y_true)


def compute_iou(y_pred, y_true, noise_cutoff=0, eps=1e-6):
    """
    Calculate an IoU-like metric for scalar values representing the intersection of the area
    under the predicted and true curves divided by the union area under the two curves.
    ---
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = (intersection + self.smooth) / (union + self.smooth)
    ---

    Parameters:
    y_true (torch.Tensor): Ground truth values, shape (batch_size, N).
    y_pred (torch.Tensor): Predicted values, shape (batch_size, N).

    Returns:
    float: IoU-like metric.
    """
    assert y_pred.shape == y_true.shape

    # cutoff noise value
    y_pred_temp = y_pred.clone().abs()
    y_true_temp = y_true.clone().abs()
    y_pred_temp[y_pred_temp < noise_cutoff] = 0
    y_true_temp[y_true_temp < noise_cutoff] = 0

    # Calculate the intersection area
    min_values = torch.min(y_true_temp, y_pred_temp)
    intersection_area = torch.trapz(min_values, dim=1)

    # Calculate the total area under the y_true curve using the trapezoidal rule
    total_area = (torch.trapz(y_true_temp, dim=1)
                  + torch.trapz(y_pred_temp, dim=1)
                  - intersection_area)

    # Compute the IoU-like metric (eps is to avoid dividing by zero)
    iou = (intersection_area + eps) / (total_area + eps)
    return iou


class MyIouLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.noise_cutoff = config.label_noise_cutoff
        self.eps = config.computation_eps

    def forward(self, y_pred, y_true):
        return 1 - compute_iou(y_pred, y_true, noise_cutoff=self.noise_cutoff, eps=self.eps).mean()


class MyIouMetric(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.noise_cutoff = config.label_noise_cutoff
        self.eps = config.computation_eps

    def forward(self, y_pred, y_true):
        return compute_iou(y_pred, y_true, noise_cutoff=self.noise_cutoff, eps=self.eps).mean()


class MyCombinedLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mae_loss = nn.L1Loss()
        self.mae_loss_lambda = config.criterion_mae_lambda

        self.mse_loss = nn.MSELoss()
        self.mse_loss_lambda = config.criterion_mse_lambda

        self.iou_loss = MyIouLoss(config)
        self.iou_loss_lambda = config.criterion_iou_lambda

    def forward(self, y_pred, y_true):
        mae_loss = self.mae_loss(y_pred, y_true) if self.mae_loss_lambda != 0 else 0
        mse_loss = self.mse_loss(y_pred, y_true) if self.mse_loss_lambda != 0 else 0
        iou_loss = self.iou_loss(y_pred, y_true) if self.iou_loss_lambda != 0 else 0
        return (mae_loss * self.mae_loss_lambda +
                mse_loss * self.mse_loss_lambda +
                iou_loss * self.iou_loss_lambda)


'''***********************************************************************'''
''' Model '''
'''***********************************************************************'''


class MyEmbeddingBlock(nn.Module):
    def __init__(self, feature_size, hidden_size, mode='default'):  #mode = 'default', 'fc'
        super().__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.enable_linear = False
        if mode == 'fc':
            self.weight = nn.Parameter(torch.randn(feature_size, hidden_size), requires_grad=True)
            self.bias = nn.Parameter(torch.randn(1, hidden_size), requires_grad=True)
            self.enable_linear = True
        else:
            self.embed = nn.ModuleList([nn.Linear(1, hidden_size) for _ in range(feature_size)])

    def forward(self, x):
        if self.enable_linear:
            B, N, _ = x.size()
            x_embed = x.expand(self.feature_size, -1, -1, -1).permute(1, 2, 3, 0)
            x_embed = x_embed.matmul(self.weight) + self.bias
            x_embed = x_embed.view(B, N, -1)
        else:
            x_embed = torch.cat([self.embed[i](x[:, :, i].unsqueeze(-1))
                                 for i in range(self.feature_size)], dim=-1)
        return x_embed


class MyInputEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_dim = 0

        self.enable_img_embed = config.enable_img_embed
        if self.enable_img_embed:
            self.hidden_dim += config.img_embed_dim * 1

        self.enable_param_embed = config.enable_param_embed
        if self.enable_param_embed:
            if not config.enable_exclude_feature:
                config.param_exclude = []
            adjusted_param_size = config.param_size - len(config.param_exclude)
            self.hidden_dim += config.param_embed_dim * adjusted_param_size
            self.param_embed = MyEmbeddingBlock(adjusted_param_size, config.param_embed_dim,
                                                config.feature_embed_option)

        self.enable_pos_embed = config.enable_pos_embed
        if self.enable_pos_embed:
            if not config.enable_exclude_feature:
                config.pos_exclude = []
            adjusted_pos_size = config.pos_size - len(config.pos_exclude)
            self.hidden_dim += config.pos_embed_dim * adjusted_pos_size
            self.pos_embed = MyEmbeddingBlock(adjusted_pos_size, config.pos_embed_dim, config.feature_embed_option)

        print(f'> MyInputEmbedding: hidden_dim = {self.hidden_dim}')
        config.embed_dim = self.hidden_dim

    def forward(self, x_img, x_param, x_pos):
        x_img = x_img if self.enable_img_embed else torch.tensor([], device=x_img.device)
        x_param = self.param_embed(x_param) if self.enable_param_embed else torch.tensor([], device=x_param.device)
        x_pos = self.pos_embed(x_pos) if self.enable_pos_embed else torch.tensor([], device=x_pos.device)
        x = torch.cat((x_img, x_param, x_pos), dim=2)
        return x  # (B, N, H_b)


from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class GNNSequenceProcessor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, batch_first=True):
        super().__init__()
        self.batch_first = batch_first

        # Create a list of GCNConv layers
        self.gcns = nn.ModuleList()
        self.gcns.append(GCNConv(input_dim, hidden_dim))  # First GCN layer

        # Add additional GCNConv layers (num_layers - 1)
        for _ in range(num_layers - 1):
            self.gcns.append(GCNConv(hidden_dim, output_dim))  # Subsequent GCN layers

    def forward(self, x):
        # x has shape (batch_size, seq_len, input_dim) when batch_first=True
        batch_size, seq_len, input_dim = x.shape

        # Step 1: Convert input tensor to graph format
        edge_index = []
        for i in range(seq_len - 1):  # connect each node to its next node
            edge_index.append([i, i + 1])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(x.device)

        # Flatten the input tensor to have shape (num_nodes, input_dim)
        node_features = x.view(-1, input_dim)

        # Create a PyTorch Geometric Data object
        data = Data(x=node_features, edge_index=edge_index).to(x.device)

        # Step 2: Apply GCN layers
        out = data.x
        for gcn in self.gcns:
            out = gcn(out, data.edge_index)  # Apply GCN layer
            out = F.relu(out)

        # Return the output node features (now of shape [batch_size * seq_len, hidden_dim])
        return data.x.view(batch_size, seq_len, -1)  # Reshape back to [batch_size, seq_len, hidden_dim]


class MySimpleModelBlock(nn.Module):
    def __init__(self, config, hidden_dim, model='None'):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.model_name = model

        self.only_one_frame = False
        self.enable_rnn = False
        self.enable_attention = False

        self.enable_residual = False
        self.enable_norm = False
        self.enable_activation = True

        if model == 'MLP':  # Multi-layer perception
            self.base_model = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim))

        elif model == 'RNN':
            self.base_model = nn.RNN(input_size=self.hidden_dim,
                       hidden_size=self.hidden_dim,
                       num_layers=1,
                       batch_first=True,
                       bidirectional=False,
                       )
            self.enable_rnn = True

        elif model == 'LSTM':
            self.base_model = nn.LSTM(input_size=self.hidden_dim,
                        hidden_size=self.hidden_dim,
                        num_layers=1,
                        batch_first=True,
                        bidirectional=False,
                        )
            self.enable_rnn = True

        elif model == 'LSTM_2':
            self.base_model =nn.LSTM(input_size=self.hidden_dim,
                        hidden_size=self.hidden_dim,
                        num_layers=2,
                        batch_first=True,
                        bidirectional=False,
                        )
            self.enable_rnn = True

        elif model == 'LSTM_3':
            self.base_model =nn.LSTM(input_size=self.hidden_dim,
                        hidden_size=self.hidden_dim,
                        num_layers=3,
                        batch_first=True,
                        bidirectional=False,
                        )
            self.enable_rnn = True

        elif model == 'BLSTM':
            self.base_model =nn.LSTM(input_size=self.hidden_dim,
                        hidden_size=self.hidden_dim // 2,
                        num_layers=1,
                        batch_first=True,
                        bidirectional=True,
                        )
            self.enable_rnn = True

        elif model == 'BLSTM_2':
            self.base_model = nn.LSTM(input_size=self.hidden_dim,
                        hidden_size=self.hidden_dim // 2,
                        num_layers=2,
                        batch_first=True,
                        bidirectional=True,
                        )
            self.enable_rnn = True

        elif model == 'SA':  # self-attention
            self.base_model = nn.MultiheadAttention(config.embed_dim,
                                      num_heads=1,
                                      # dropout=config.dropout if config.enable_dropout else 0,
                                      batch_first=True)
            self.enable_attention = True

        elif model == 'MHSA-4':  # multi-head self-attention
            self.base_model = nn.MultiheadAttention(config.embed_dim,
                                      num_heads=4,
                                      # dropout=config.dropout if config.enable_dropout else 0,
                                      batch_first=True)
            self.enable_attention = True

        elif model == 'MHSA-4-Res':  # multi-head self-attention
            self.base_model = nn.MultiheadAttention(config.embed_dim,
                                                    num_heads=4,
                                                    # dropout=config.dropout if config.enable_dropout else 0,
                                                    batch_first=True)
            self.enable_attention = True
            self.enable_residual = True

        elif model == 'MHSA-4-Res-Norm':  # multi-head self-attention
            self.base_model = nn.MultiheadAttention(config.embed_dim,
                                                    num_heads=4,
                                                    # dropout=config.dropout if config.enable_dropout else 0,
                                                    batch_first=True)
            self.ln = nn.LayerNorm(config.embed_dim)
            self.enable_attention = True
            self.enable_residual = True
            self.enable_norm = True

        elif model == 'MHSA-8':  # multi-head self-attention
            self.base_model = nn.MultiheadAttention(config.embed_dim,
                                      num_heads=8,
                                      # dropout=config.dropout if config.enable_dropout else 0,
                                      batch_first=True)
            self.enable_attention = True

        elif model == 'GCN-1':  # graph neural networks
            self.base_model = GNNSequenceProcessor(input_dim=self.hidden_dim,
                                                   hidden_dim=self.hidden_dim,
                                                   output_dim=self.hidden_dim,
                                                   num_layers=1,
                                                   )
            self.enable_activation = False

        elif model == 'GCN-2':  # graph neural networks
            self.base_model = GNNSequenceProcessor(input_dim=self.hidden_dim,
                                                   hidden_dim=self.hidden_dim,
                                                   output_dim=self.hidden_dim,
                                                   num_layers=2, )
            self.enable_activation = False

        elif model == 'GCN-3':  # graph neural networks
            self.base_model = GNNSequenceProcessor(input_dim=self.hidden_dim,
                                                   hidden_dim=self.hidden_dim,
                                                   output_dim=self.hidden_dim,
                                                   num_layers=3, )
            self.enable_activation = False

        else:
            raise RuntimeError(f'Unknown model: {model}')

    def forward(self, x):
        if self.enable_residual:
            x_res = x

        if self.enable_rnn:
            x, _ = self.base_model(x, hx=None)
        elif self.enable_attention:
            x = self.base_model(x, x, x, need_weights=False)[0]
        else:
            x = self.base_model(x)

        if self.enable_residual:
            x = x_res + x

        if self.enable_norm:
            x = self.ln(x)

        if self.enable_activation:
            x = F.relu(x)

        return x




class MyFeatureAttnBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.enable_residual_gamma = config.enable_residual_gamma
        if self.enable_residual_gamma:
            self.gamma = nn.Parameter(torch.zeros(1))

        # feature wise attn
        self.attn = nn.MultiheadAttention(config.n_seq_enc_total,
                                          num_heads=config.encoder_num_heads,
                                          dropout=config.dropout if config.enable_dropout else 0,
                                          batch_first=True)

        # save attention
        self.enable_save_attention = config.enable_save_attention
        if self.enable_save_attention:
            self.attn_map = None

        # layer norm
        self.enable_layer_norm = config.enable_layer_norm
        if self.enable_layer_norm:
            self.ln = nn.LayerNorm(config.n_seq_enc_total)

    def forward(self, x):
        x = x.transpose(1, 2)
        if self.enable_save_attention:
            attn_out, self.attn_map = self.attn(x, x, x, need_weights=True)
        else:
            attn_out = self.attn(x, x, x, need_weights=False)[0]

        if self.enable_residual_gamma:
            x = self.gamma * attn_out + x
        else:
            x = x + attn_out  # (B, N, H_c)

        if self.enable_layer_norm:
            x = self.ln(x)

        x = x.transpose(1, 2)
        return x


class MyTemporalAttnBlock(nn.Module):
    def __init__(self, config, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.enable_residual_gamma = config.enable_residual_gamma
        if self.enable_residual_gamma:
            self.gamma = nn.Parameter(torch.zeros(1))

        # temporal wise attn
        self.attn = nn.MultiheadAttention(self.hidden_dim,
                                          num_heads=config.encoder_num_heads,
                                          dropout=config.dropout if config.enable_dropout else 0,
                                          batch_first=True)
        # save attention
        self.enable_save_attention = config.enable_save_attention
        if self.enable_save_attention:
            self.attn_map = None

        # layer norm
        self.enable_layer_norm = config.enable_layer_norm
        if self.enable_layer_norm:
            self.ln = nn.LayerNorm(config.embed_dim)

    def forward(self, x):
        if self.enable_save_attention:
            attn_out, self.attn_map = self.attn(x, x, x, need_weights=True)
        else:
            attn_out = self.attn(x, x, x, need_weights=False)[0]

        if self.enable_residual_gamma:
            x = self.gamma * attn_out + x
        else:
            x = x + attn_out  # (B, N, H_c)

        if self.enable_layer_norm:
            x = self.ln(x)

        return x


class MyRnnBlock(nn.Module):
    def __init__(self, config, hidden_dim, model='LSTM', bidirectional=True, ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # temporal wise attn
        if model == 'LSTM':
            self.base_model = nn.LSTM(input_size=self.hidden_dim,
                                      hidden_size=self.hidden_dim // 2 if bidirectional else self.hidden_dim,
                                      num_layers=1,
                                      batch_first=True,
                                      bidirectional=bidirectional,
                                      )
        elif model == 'RNN':
            self.base_model = nn.LSTM(input_size=self.hidden_dim,
                                      hidden_size=self.hidden_dim // 2 if bidirectional else self.hidden_dim,
                                      num_layers=1,
                                      batch_first=True,
                                      bidirectional=bidirectional,
                                      )
        else:
            raise RuntimeError(f'Unknown model: {model}')

        self.hx = None

        # layer norm
        self.enable_layer_norm = config.enable_layer_norm
        if self.enable_layer_norm:
            self.ln = nn.LayerNorm(self.hidden_dim)

        print(f'> MyBiLSTMBlock param: {get_model_parameter_num(self)}')
        print(f'    > lstm param: {get_model_parameter_num(self.base_model)}')

    def forward(self, x, reset_hx=True):
        if reset_hx:
            self.hx = None

        x, self.hx = self.base_model(x, hx=self.hx)

        if self.enable_layer_norm:
            x = self.ln(x)

        return x


class MyFeedForwardBlock(nn.Module):
    def __init__(self, config, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.enable_dropout = config.enable_dropout
        if self.enable_dropout:
            self.dropout = nn.Dropout(config.dropout)

        self.enable_residual_gamma = config.enable_residual_gamma
        if self.enable_residual_gamma:
            self.gamma = nn.Parameter(torch.zeros(1))

        self.enable_layer_norm = config.enable_layer_norm
        if self.enable_layer_norm:
            self.ln = nn.LayerNorm(self.hidden_dim)

    def forward(self, x):
        if self.enable_dropout:
            x = self.dropout(x)

        if self.enable_residual_gamma:
            x = x + self.gamma * self.fc2(self.relu(self.fc1(x)))
        else:
            x = x + self.fc2(self.relu(self.fc1(x)))

        if self.enable_layer_norm:
            x = self.ln(x)
        return x


class MyEncoder(nn.Module):
    def __init__(self, config, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        layers = []
        if config.model == 'STEN_GP_FFD_BLSTM':
            for _ in range(config.encoder_layer_size):
                layers.append(MyFeedForwardBlock(config, self.hidden_dim))
                layers.append(MyRnnBlock(config, self.hidden_dim, model='LSTM', bidirectional=True))
        elif config.model == 'STEN_GP_BLSTM_FFD':
            for _ in range(config.encoder_layer_size):
                layers.append(MyRnnBlock(config, self.hidden_dim, model='LSTM', bidirectional=True))
                layers.append(MyFeedForwardBlock(config, self.hidden_dim))
        elif config.model == 'STEN_GP_GCN_2_FFD':
            for _ in range(config.encoder_layer_size):
                layers.append(MySimpleModelBlock(config, self.hidden_dim, model='GCN-2'))
                layers.append(MyFeedForwardBlock(config, self.hidden_dim))
        elif config.model == 'STEN_GP_FFD_TA':
            for _ in range(config.encoder_layer_size):
                layers.append(MyFeedForwardBlock(config, self.hidden_dim))
                layers.append(MyTemporalAttnBlock(config, self.hidden_dim))
        elif config.model == 'STEN_GP_TA_FFD':
            for _ in range(config.encoder_layer_size):
                layers.append(MyTemporalAttnBlock(config, self.hidden_dim))
                layers.append(MyFeedForwardBlock(config, self.hidden_dim))
        elif config.model == 'STEN_GP_FA_TA':
            for _ in range(config.encoder_layer_size):
                layers.append(MyFeatureAttnBlock(config))
                layers.append(MyTemporalAttnBlock(config, self.hidden_dim))
        elif config.model == 'STEN_GP_MHSA_4_Res_FFD':
            for _ in range(config.encoder_layer_size):
                layers.append(MySimpleModelBlock(config, self.hidden_dim, model='MHSA-4-Res'))
                layers.append(MyFeedForwardBlock(config, self.hidden_dim))
        elif config.model == 'STEN_GP_MHSA_4_Res_Norm_FFD':
            for _ in range(config.encoder_layer_size):
                layers.append(MySimpleModelBlock(config, self.hidden_dim, model='MHSA-4-Res-Norm'))
                layers.append(MyFeedForwardBlock(config, self.hidden_dim))
        elif config.model == 'STEN_GP_LSTM_2_FFD':
            for _ in range(config.encoder_layer_size):
                layers.append(MySimpleModelBlock(config, self.hidden_dim, model='LSTM_2'))
                layers.append(MyFeedForwardBlock(config, self.hidden_dim))
        elif config.model == 'STEN_GP_FFD':
            for _ in range(config.encoder_layer_size):
                layers.append(MyFeedForwardBlock(config, self.hidden_dim))
        elif config.model == 'STEN_GP_Simple_MLP':
            layers.append(MySimpleModelBlock(config, self.hidden_dim, model='MLP'))
        elif config.model == 'STEN_GP_Simple_RNN':
            layers.append(MySimpleModelBlock(config, self.hidden_dim, model='RNN'))
        elif config.model == 'STEN_GP_Simple_LSTM':
            layers.append(MySimpleModelBlock(config, self.hidden_dim, model='LSTM'))
        elif config.model == 'STEN_GP_Simple_LSTM_2':
            layers.append(MySimpleModelBlock(config, self.hidden_dim, model='LSTM_2'))
        elif config.model == 'STEN_GP_Simple_LSTM_2_Simple_MLP':
            layers.append(MySimpleModelBlock(config, self.hidden_dim, model='LSTM_2'))
            layers.append(MySimpleModelBlock(config, self.hidden_dim, model='MLP'))
        elif config.model == 'STEN_GP_Simple_LSTM_3':
            layers.append(MySimpleModelBlock(config, self.hidden_dim, model='LSTM_3'))
        elif config.model == 'STEN_GP_Simple_BLSTM':
            layers.append(MySimpleModelBlock(config, self.hidden_dim, model='BLSTM'))
        elif config.model == 'STEN_GP_Simple_BLSTM_2':
            layers.append(MySimpleModelBlock(config, self.hidden_dim, model='BLSTM_2'))
        elif config.model == 'STEN_GP_Simple_BLSTM_3':
            layers.append(MySimpleModelBlock(config, self.hidden_dim, model='BLSTM_3'))
        elif config.model == 'STEN_GP_Simple_SA':
            layers.append(MySimpleModelBlock(config, self.hidden_dim, model='SA'))
        elif config.model == 'STEN_GP_Simple_MHSA':
            layers.append(MySimpleModelBlock(config, self.hidden_dim, model='MHSA-4'))
        elif config.model == 'STEN_GP_2_Simple_MHSA':
            layers.append(MySimpleModelBlock(config, self.hidden_dim, model='MHSA-4'))
            layers.append(MySimpleModelBlock(config, self.hidden_dim, model='MHSA-4'))
        elif config.model == 'STEN_GP_Simple_MHSA_8':
            layers.append(MySimpleModelBlock(config, self.hidden_dim, model='MHSA-8'))
        elif config.model == 'STEN_GP_Simple_GCN_1':
            layers.append(MySimpleModelBlock(config, self.hidden_dim, model='GCN-1'))
        elif config.model == 'STEN_GP_Simple_GCN_2':
            layers.append(MySimpleModelBlock(config, self.hidden_dim, model='GCN-2'))
        elif config.model == 'STEN_GP_Simple_GCN_3':
            layers.append(MySimpleModelBlock(config, self.hidden_dim, model='GCN-3'))
        else:
            raise RuntimeError(f'Unknown model: {config.model}')

        self.encoder = nn.Sequential(*layers)

        print(f'> MyEncoder param: {get_model_parameter_num(self)}')
        for i_layer, layer in enumerate(layers):
            print(f'    > Layer: {i_layer + 1}: {get_model_parameter_num(layer)}')

    def forward(self, x):
        return self.encoder(x)


class MyDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_seq_before = config.n_seq_enc_look_back

        self.decoder = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout) if config.enable_dropout else nn.Identity(),

            nn.Linear(config.embed_dim // 2, config.embed_dim // 4),
            nn.ReLU(),
            nn.Linear(config.embed_dim // 4, config.output_size),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x[:, self.n_seq_before, :]
        x = self.decoder(x)
        return x


class MyDecoderPlus(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_seq_before = config.n_seq_enc_look_back

        # embedding
        self.embed = MyEmbeddingBlock(config.output_size, config.output_embed_dim)

        # st layer
        self.output_latent_dim = config.output_size * config.output_embed_dim
        self.st_layer_1 = MyFeedForwardBlock(config, self.output_latent_dim)
        self.st_layer_2 = MyRnnBlock(config, self.output_latent_dim, bidirectional=False)
        self.hx = None

        # output
        self.final_in_dim = config.embed_dim + self.output_latent_dim
        self.final = nn.Sequential(
            nn.Linear(self.final_in_dim, self.final_in_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout) if config.enable_dropout else nn.Identity(),

            nn.Linear(self.final_in_dim // 2, self.final_in_dim // 4),
            nn.ReLU(),
            nn.Linear(self.final_in_dim // 4, config.output_size),
            nn.ReLU(),
        )

    def forward(self, x_latent, y_prev, reset_hx=False):
        y_embed = self.embed(y_prev)
        y_latent = self.st_layer_1(y_embed)
        y_latent = self.st_layer_2(y_latent, reset_hx=reset_hx)[:, -1, :]

        x_latent = x_latent[:, self.n_seq_before, :]
        x = torch.cat((x_latent, y_latent), dim=-1)
        x = self.final(x)
        return x


class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = MyInputEmbedding(config)
        self.hidden_dim = self.embedding.hidden_dim
        self.encoder = MyEncoder(config, self.hidden_dim)
        if 'decoder_option' in config and config.decoder_option == 'transformer':
            self.decoder = MyDecoderPlus(config)
            self.enable_auto_regression = True
        else:
            self.decoder = MyDecoder(config)
            self.enable_auto_regression = False
        self.timing_on = False

    def forward(self, x_img, x_param, x_pos, y_prev=None, reset_dec_hx=False):
        if self.timing_on:
            LAYER_TIMING_NUM = 10000
            t_start = time.time()
            for i_time in range(LAYER_TIMING_NUM):
                x = self.embedding(x_img, x_param, x_pos)  # (B, N, H_b)
            t_emb = (time.time() - t_start) / LAYER_TIMING_NUM

            x_temp = x
            t_start = time.time()
            for i_time in range(LAYER_TIMING_NUM):
                x = self.encoder(x_temp)  # (B, N, H_c)
            t_enc = (time.time() - t_start) / LAYER_TIMING_NUM

            x_temp = x
            t_start = time.time()
            for i_time in range(LAYER_TIMING_NUM):
                if self.enable_auto_regression:
                    x = self.decoder(x_temp, y_prev, reset_hx=reset_dec_hx)  # (B, N, output_size)
                else:
                    x = self.decoder(x_temp)
            t_dec = (time.time() - t_start) / LAYER_TIMING_NUM

            t_total = t_emb + t_enc + t_dec
            self.t_emb = t_emb
            self.t_enc = t_enc
            self.t_dec = t_dec
            self.t_total = t_total
            print(f'> emb:{t_emb * 1000}ms, enc:{t_enc * 1000}ms, dec:{t_dec * 1000}ms, total:{t_total * 1000}ms')
            return x
        else:
            x = self.embedding(x_img, x_param, x_pos)  # (B, N, H_b)
            x = self.encoder(x)  # (B, N, H_c)
            if self.enable_auto_regression:
                x = self.decoder(x, y_prev, reset_hx=reset_dec_hx)  # (B, N, output_size)
            else:
                x = self.decoder(x)
            return x


'''***********************************************************************'''
''' Adaptor '''
'''***********************************************************************'''


class MyAdaptor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.methods = {'GMM3': self.compute_gmm,
                        'Fourier3': self.compute_fourier,
                        'Sigmoid6': self.compute_mirror_sigmoid,
                        }

        self.adaptor_option = config.adaptor_option
        if self.adaptor_option not in self.methods:
            raise RuntimeError(f'Adaptor option {self.adaptor_option} is not supported')
        self.method = self.methods[self.adaptor_option]

        self.component_size = config.adaptor_component_size
        self.label_size = config.label_size
        self.x = torch.tensor(range(self.label_size), dtype=torch.float32, device=config.device)
        self.x = self.x.view(1, 1, -1)  # Reshape for broadcasting

    def compute_gmm(self, x, params):
        # param[:, :, 0]: height
        # param[:, :, 1]: steepness
        # param[:, :, 2]: x shift
        # steepness = torch.clamp(params[:, :, 1], max=50).unsqueeze(-1)  # Prevent overflow in exp
        y = params[:, :, 0].unsqueeze(-1) * torch.exp(
            -params[:, :, 1].unsqueeze(-1) * (x - params[:, :, 2].unsqueeze(-1)) ** 2)
        y = torch.nan_to_num(y, nan=0.0)  # Replace NaNs with 0
        return y

    def compute_fourier(self, x, param):
        # param[0]: height
        # param[1]: frequency
        # param[2]: x shift
        y = param[0] * torch.sin(param[1] * (x - param[2]))
        y = torch.nan_to_num(y, nan=0.0)  # Replace NaNs with 0
        return y

    def compute_mirror_sigmoid(self, x, param):
        # param[0]: central position
        # param[1]: height
        # param[2]: steepness left
        # param[3]: x shift left
        # param[4]: steepness right
        # param[5]: x shift right

        steepness_left = torch.clamp(param[2], max=50)  # Prevent overflow in exp
        y_left = param[1] / (1 + torch.exp(steepness_left * (x - param[3])))
        y_left[x < param[0]] = 0

        steepness_right = torch.clamp(param[4], max=50)  # Prevent overflow in exp
        y_right = param[1] / (1 + torch.exp(steepness_right * (x - param[5])))
        y_right[x > param[0]] = 0

        y = y_left + y_right
        y = torch.nan_to_num(y, nan=0.0)  # Replace NaNs with 0
        return y

    def compute(self, params):
        params = params.view(params.size(0), self.component_size, 3)
        result = self.method(self.x, params)
        result = result.sum(dim=1)  # Sum over the component axis
        return result


'''***********************************************************************'''
''' The main function '''
'''***********************************************************************'''


def get_model(config):
    # model
    model = MyModel(config)
    model = model.to(config.device)
    print(f'> model name: {config.model}, param num: {get_model_parameter_num(model)}')

    if 'embed_require_grad' in config and config.embed_require_grad is False:
        freeze_parameters(model.embedding)
    if 'encoder_require_grad' in config and config.encoder_require_grad is False:
        freeze_parameters(model.encoder)
    if 'decoder_require_grad' in config and config.decoder_require_grad is False:
        freeze_parameters(model.decoder)

    # auto parallel in multiple gpu
    if config.enable_gpu:
        if 'enable_ddp' in config and config.enable_ddp:
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[config.ddp_local_rank],
                                                              output_device=config.ddp_local_rank, )
        else:
            model = nn.DataParallel(model, device_ids=[config.dp_core_idx])

    # Init all params, ex using kaiming mothod
    if config.enable_param_init:
        model.apply(init_model_weights)

    # Load pre-trained params from file
    if 'enable_preload_param' in config and config.enable_preload_param is True:
        preload_param = torch.load(config.preload_param_path)
        preload_state_dict = {}

        if config.embed_preload_param is True:
            temp_state_dict = {k: v for k, v in preload_param.items() if k.startswith('module.embedding.')}
            preload_state_dict.update(temp_state_dict)

        if config.encoder_preload_param is True:
            temp_state_dict = {k: v for k, v in preload_param.items() if k.startswith('module.encoder.')}
            preload_state_dict.update(temp_state_dict)

        if config.decoder_preload_param is True:
            temp_state_dict = {k: v for k, v in preload_param.items() if k.startswith('module.decoder.')}
            preload_state_dict.update(temp_state_dict)

        model.load_state_dict(preload_state_dict, strict=False)

    # adaptor/reconstructor
    adaptor = None
    if config.enable_adaptor:
        adaptor = MyAdaptor(config).to(config.device)

    # criterion
    criterion = None
    if config.criterion_option == 'mse':
        criterion = nn.MSELoss()
    elif config.criterion_option == 'iou':
        criterion = MyIouLoss(config)
    elif config.criterion_option == 'mse_iou':
        criterion = MyCombinedLoss(config)
    elif config.criterion_option == 'mae_mse_iou':
        criterion = MyCombinedLoss(config)
    else:
        raise RuntimeError(f'Criterion option {config.criterion_option} is not supported')

    # metric
    metric = None
    if config.metric_option == 'mse':
        metric = nn.MSELoss()
    elif config.metric_option == 'rmse':
        metric = MyRmseMetric(config)
    elif config.metric_option == 'miou':
        metric = MyIouMetric(config)
    elif config.metric_option == 'mape':
        metric = MyMapeMetric(config)
    elif config.metric_option == 'mapa':
        metric = MyMapaMetric(config)
    elif config.metric_option == 'mae':
        metric = MyMaeMetric(config)
    else:
        raise RuntimeError(f'Metric option {config.metric_option} is not supported')

    # optimizer
    optimizer = None
    if config.optimizer_option == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.lr,
                                     weight_decay=config.wd if config.enable_weight_decay else 0)
    elif config.optimizer_option == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=config.lr,
                                      weight_decay=config.wd if config.enable_weight_decay else 0)
    elif config.optimizer_option == 'adam_Vaswani_2017':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=config.lr,
                                      betas=(0.9, 0.98),
                                      weight_decay=config.wd if config.enable_weight_decay else 0)
    else:
        raise RuntimeError(f'optimizer option {config.optimizer_option} is not supported')

    # scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_gamma)

    return model, adaptor, criterion, metric, optimizer, scheduler


if __name__ == '__main__':
    ENABLE_PROFILING = False
    ENABLE_LAYER_TIMING = True
    ENABLE_MODEL_SUMMARY = True

    config = load_config()
    setup_local_device(config)

    ''' I/O test '''

    model_names = [
        'STEN_GP_Simple_MLP',
        'STEN_GP_Simple_RNN',
        'STEN_GP_Simple_LSTM',
        'STEN_GP_Simple_LSTM_2',
        'STEN_GP_Simple_LSTM_2_Simple_MLP',
        'STEN_GP_Simple_LSTM_3',
        'STEN_GP_Simple_BLSTM',
        'STEN_GP_Simple_BLSTM_2',
        'STEN_GP_Simple_SA',
        'STEN_GP_Simple_MHSA',
        'STEN_GP_Simple_MHSA_8',
        'STEN_GP_2_Simple_MHSA',
        'STEN_GP_Simple_GCN_1',
        'STEN_GP_Simple_GCN_2',
        'STEN_GP_Simple_GCN_3',
        'STEN_GP_GCN_2_FFD',
        'STEN_GP_MHSA_4_Res_FFD',
        'STEN_GP_MHSA_4_Res_Norm_FFD',
        'STEN_GP_LSTM_2_FFD',
        'STEN_GP_BLSTM_FFD',
    ]
    # model_names = [
    #     'STEN_GP_Simple_LSTM_2',
    #     'STEN_GP_Simple_BLSTM_2',
    # ]
    # model_names = ['STEN_GP_Simple_MHSA_8']
    total_params_list_list = []
    timing_history_list = []
    # INPUT_LEN_LIST = [25*k for k in range(1, 31)]
    INPUT_LEN_LIST = [200]
    for INPUT_LEN in INPUT_LEN_LIST:
        # H_list = [4, 6, 8]
        H_list = [6]
        for H in H_list:
            ''' Design embedding size '''
            print('>' * 100)
            print(f"\n> config embedding design...")
            total_params_list = []
            stl_params_list = []
            stl_params_pct = []
            config.batch_size = 1
            config.param_embed_dim = H
            config.pos_embed_dim = H
            config.embed_dim = (config.img_embed_dim +
                                config.param_embed_dim * config.param_size +
                                config.pos_embed_dim * config.pos_size)
            config.n_seq_enc_look_back = INPUT_LEN
            config.n_seq_enc_look_ahead = 0
            config.n_seq_enc_total = INPUT_LEN + 1
            config.enable_preload_param = False
            print(f"> config.param_embed_dim = {H}, config.pos_embed_dim = {H}")
            print(f"> config.embed_dim = {config.embed_dim}")
            print(f"> n_seq_enc_total = {config.n_seq_enc_total}")

            ''' Design input '''
            print(f"\n> input design...")
            B = config.batch_size  # batch size
            N = config.n_seq_enc_total

            x_img = torch.randn(B, N, config.img_embed_dim).to(config.device)
            x_param = torch.randn(B, N, config.param_size).to(config.device)
            x_pos = torch.randn(B, N, config.pos_size).to(config.device)
            y_prev = torch.randn(B, config.n_seq_enc_look_back, config.output_size).to(config.device)
            y = torch.randn(B, config.output_size).to(config.device)

            print("> device: ", config.device)
            print("> x_img shape: ", x_img.shape)
            print("> x_param shape: ", x_param.shape)
            print("> x_pos shape: ", x_pos.shape)
            print("> y shape: ", y.shape)

            for model_name in model_names:
                print('')
                print('>' * 50)
                print('> model: ', model_name)
                config.model = model_name
                model, _, _, _, _, _ = get_model(config)
                total_params = get_model_parameter_num(model.module)
                total_params_list.append(total_params)
                print('> model param size: ', total_params)
                print('> model neuron num: ', get_model_neuron_num(model))
                model.module.timing_on = ENABLE_LAYER_TIMING  # print timing of each layer
                if ENABLE_LAYER_TIMING:
                    model.eval()
                    print('> layer timing on...')
                if config.decoder_option == 'transformer':
                    y = model(x_img, x_param, x_pos, y_prev)
                else:
                    y = model(x_img, x_param, x_pos)
                print("> output shape", y.shape)

                if ENABLE_LAYER_TIMING:
                    timing_history_list.append([config.n_seq_enc_total,
                                                model.module.t_emb, model.module.t_enc, model.module.t_dec,
                                                model.module.t_total])
                    model.module.timing_on = False  # disable for following analysis

                if ENABLE_MODEL_SUMMARY:
                    from torchinfo import summary

                    if config.decoder_option == 'transformer':
                        summary(model, input_size=[x_img.shape, x_param.shape, x_pos.shape, y_prev.shape])
                    else:
                        summary(model, input_size=[x_img.shape, x_param.shape, x_pos.shape])

                if ENABLE_PROFILING:
                    '''Profile the forward pass'''
                    import torch.profiler

                    num_iterations = 200
                    with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ],
                            schedule=torch.profiler.schedule(wait=1, warmup=1, active=num_iterations),
                            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/my_embedding_profile'),
                            record_shapes=True,
                            profile_memory=True,
                            with_stack=True
                    ) as prof:
                        time_start = time.time()
                        for _ in range(num_iterations):
                            output = model(x_img, x_param, x_pos)
                            prof.step()

                    print(f"total elapsed time: {(time.time() - time_start) * 1000:.3f}ms")
                    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

            total_params_list_list.append(total_params_list)

    print('> models:')
    print(model_names)

    print('> total param')
    print(total_params_list_list)

    print(f"> timing history:")
    print(timing_history_list)
