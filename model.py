import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from util import *

# class LayerBiLstm(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#
#         self.H = config.embed_dim
#         self.lstm = nn.LSTM(input_size=self.H,
#                             hidden_size=self.H // 2,
#                             num_layers=config.lstm_layer_num,
#                             bidirectional=True,
#                             batch_first=True)
#
#     def forward(self, x):  # x: (B, N, H)
#         return self.lstm(x, hx=None)[0]  # (B, N, H)


class MyEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        N = config.n_seq_total
        self.H_param = config.param_size
        self.H_pos = config.pos_size
        self.param_embed = nn.ModuleList([nn.Linear(1, config.param_embed_dim) for _ in range(self.H_param)])
        self.pos_embed = nn.ModuleList([nn.Linear(1, config.pos_embed_dim) for _ in range(self.H_pos)])
        self.embed_dim = (config.img_embed_dim +
                          config.param_embed_dim * config.param_size +
                          config.pos_embed_dim * config.pos_size)
        if self.embed_dim != config.embed_dim:
            raise RuntimeError(f"self.embed_dim({self.embed_dim}) != config.embed_dim({config.embed_dim})")

    def forward(self, x_img, x_param, x_pos):
        B = x_img.size(0)
        N = x_img.size(1)
        x_param = torch.cat([self.param_embed[i](x_param[:, :, i].unsqueeze(-1)) for i in range(self.H_param)], dim=-1)
        x_pos = torch.cat([self.pos_embed[i](x_pos[:, :, i].unsqueeze(-1)) for i in range(self.H_pos)], dim=-1)
        x = torch.cat((x_img, x_param, x_pos), dim=2)
        return x  # (B, N, H_b)


class MyFeatureAttnBlock(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()

        self.enable_residual_gamma = config.enable_residual_gamma
        if self.enable_residual_gamma:
            self.gamma = nn.Parameter(torch.zeros(1))

        # feature wise attn
        self.attn = nn.MultiheadAttention(config.n_seq_total,
                                          num_heads=config.encoder_num_heads,
                                          dropout=config.dropout if config.enable_dropout else 0,
                                          batch_first=True)

        self.enable_layer_norm = config.enable_layer_norm
        if self.enable_layer_norm:
            self.ln = nn.LayerNorm(config.n_seq_total)

    def forward(self, x):
        x = x.transpose(1, 2)
        if self.enable_residual_gamma:
            x = self.gamma * self.attn(x, x, x)[0] + x
        else:
            x = self.attn(x, x, x)[0]  # (B, N, H_c)

        if self.enable_layer_norm:
            x = self.ln(x)

        x = x.transpose(1, 2)
        return x


class MyTemporalAttnBlock(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()

        self.enable_residual_gamma = config.enable_residual_gamma
        if self.enable_residual_gamma:
            self.gamma = nn.Parameter(torch.zeros(1))

        # temporal wise attn
        self.attn = nn.MultiheadAttention(config.embed_dim,
                                          num_heads=config.encoder_num_heads,
                                          dropout=config.dropout if config.enable_dropout else 0,
                                          batch_first=True)

        self.enable_layer_norm = config.enable_layer_norm
        if self.enable_layer_norm:
            self.ln = nn.LayerNorm(config.embed_dim)

    def forward(self, x):
        if self.enable_residual_gamma:
            x = self.gamma * self.attn(x, x, x)[0] + x
        else:
            x = self.attn(x, x, x)[0]  # (B, N, H_c)

        if self.enable_layer_norm:
            x = self.ln(x)

        return x


class MyFeedForwardBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc1 = nn.Linear(config.embed_dim, config.embed_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config.embed_dim, config.embed_dim)

        self.enable_dropout = config.enable_dropout
        if self.enable_dropout:
            self.dropout = nn.Dropout(config.dropout)

        self.enable_residual_gamma = config.enable_residual_gamma
        if self.enable_residual_gamma:
            self.gamma = nn.Parameter(torch.zeros(1))

        self.enable_layer_norm = config.enable_layer_norm
        if self.enable_layer_norm:
            self.ln = nn.LayerNorm(config.embed_dim)

    def forward(self, x):
        if self.enable_dropout:
            x = self.dropout(x)

        if self.enable_residual_gamma:
            x = x + self.gamma * self.fc2(self.relu(self.fc1(x)))
        else:
            x = self.fc2(self.relu(self.fc1(x)))

        if self.enable_layer_norm:
            x = self.ln(x)
        return x


class MyEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        layers = []
        for _ in range(config.encoder_layer_size):
            layers.append(MyFeatureAttnBlock(config))
            layers.append(MyTemporalAttnBlock(config))
            layers.append(MyFeedForwardBlock(config))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class MyDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_seq_before = config.n_seq_before

        self.decoder = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout() if config.enable_dropout else nn.Identity(),

            nn.Linear(config.embed_dim // 2, config.embed_dim // 4),
            nn.ReLU(),
            nn.Linear(config.embed_dim // 4, config.output_size),
            nn.ReLU(),
        )

        self.save_activation = config.enable_save_activation
        self.activation = None  # record if each neuron is activated

    def forward(self, x):
        x = x[:, self.n_seq_before, :]
        x = self.decoder(x)

        if self.save_activation:
            self.activation = x

        return x


class OurModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = MyEmbedding(config)
        self.encoder = MyEncoder(config)
        self.decoder = MyDecoder(config)

    def forward(self, x_img, x_param, x_pos):
        x = self.embedding(x_img, x_param, x_pos)  # (B, N, H_b)
        x = self.encoder(x)  # (B, N, H_c)
        x = self.decoder(x)  # (B, N, output_size)
        return x


def get_model(config):
    model = OurModel(config).to(config.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.lr,
                                 weight_decay=config.wd if config.enable_weight_decay else 0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_gamma)
    return model, criterion, optimizer, scheduler


if __name__ == '__main__':
    config = load_config()

    def get_total_parameter_num(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    def get_total_neuron_num(model):
        total_neurons = 0

        def count_neurons(layer):
            nonlocal total_neurons
            if isinstance(layer, nn.Linear):
                total_neurons += layer.out_features
            elif isinstance(layer, nn.Conv2d):
                out_channels = layer.out_channels
                kernel_size = layer.kernel_size[0] * layer.kernel_size[1]  # Assuming square kernels
                out_neurons = out_channels * kernel_size
                total_neurons += out_neurons
            elif isinstance(layer, (nn.LSTM, nn.GRU, nn.RNN)):
                num_directions = 2 if layer.bidirectional else 1
                total_neurons += layer.hidden_size * num_directions
            elif isinstance(layer, nn.Module):
                for sublayer in layer.children():
                    count_neurons(sublayer)

        count_neurons(model)
        return total_neurons


    ''' I/O test '''
    model_names = ['STEN-GP']
    total_params_list_list = []
    H_list = [4, 8, 16, 32]
    for H in H_list:
        ''' Design embedding size '''
        print(f"\n> config embedding design...")
        total_params_list = []
        stl_params_list = []
        stl_params_pct = []
        config.param_embed_dim = H
        config.pos_embed_dim = H
        config.embed_dim = (config.img_embed_dim +
                            config.param_embed_dim * config.param_size +
                            config.pos_embed_dim * config.pos_size)
        print(f"> config.param_embed_dim = {H}, config.pos_embed_dim = {H}")
        print(f"> config.embed_dim = {config.embed_dim}")

        ''' Design input '''
        print(f"\n> input design...")
        B = config.batch_size  # batch size
        N = config.n_seq_total

        x_img = torch.randn(B, N, config.img_embed_dim).to(config.device)
        x_param = torch.randn(B, N, config.param_size).to(config.device)
        x_pos = torch.randn(B, N, config.pos_size).to(config.device)
        y = torch.randn(B, config.output_size).to(config.device)

        print("> device: ", config.device)
        print("> x_img shape: ", x_img.shape)
        print("> x_param shape: ", x_param.shape)
        print("> x_pos shape: ", x_pos.shape)
        print("> y shape: ", y.shape)

        print(f"\n> model design...")
        for model_name in model_names:
            print('\n')
            print('>' * 50)
            print('> model: ', model_name)
            config.model = model_name
            model, _, _, _ = get_model(config)
            total_params = get_total_parameter_num(model)
            total_params_list.append(total_params)
            print('> model param size: ', total_params)
            print('> model neuron num: ', get_total_neuron_num(model))
            y = model(x_img, x_param, x_pos)
            print("> output shape", y.shape)

            from torchinfo import summary
            summary(model, input_size=[x_img.shape, x_param.shape, x_pos.shape])

        total_params_list_list.append(total_params_list)

    import re
    def format_number(number):
        return re.sub(r'(?<!^)(?=(\d{3})+$)', ',', str(number))

    for i_list, _param_list in enumerate(total_params_list_list):
        print(f"> Param list for H = {H_list[i_list]}: ",
              [format_number(num) for num in _param_list])