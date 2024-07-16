import torch
import torch.nn as nn
from torchvision.models import resnet18
from util import *


class LayerImageConvolution(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = config.device
        self.H = config.fc_hidden_size

        self.enable_resnet_in_model = config.enable_resnet_in_model
        if self.enable_resnet_in_model:
            self.resnet = resnet18(pretrained=True)

            # Remove the last layer (the fully connected layer)
            self.resnet_fc_size = self.resnet.fc.in_features
            self.resnet.fc = nn.Identity()

            # Freeze all the parameters in resnet
            for param in self.resnet.parameters():
                param.requires_grad = False

        else:
            self.resnet_fc_size = config.resnet_fc_size

        self.fc = nn.Linear(self.resnet_fc_size, out_features=self.H)

        # only take single image as input
        if config.model.startswith('CF1'):
            self.use_one_image = True
            self.n_seq_before = config.n_seq_before
        else:
            self.use_one_image = False

    def forward(self, x):
        if self.enable_resnet_in_model:  # x: (B, N, C, H, W)
            x_img = torch.empty(x.size(0), 0, self.resnet_fc_size).to(self.device)  # (B, 0, R)
            for t in range(x.size(1)):
                with torch.no_grad():
                    x_resnet = self.resnet(x[:, t, :, :, :]).squeeze()  # (B, R)
                    x_img = torch.cat((x_img, x_resnet.unsqueeze(1)), dim=1)  # (B, N, R)

            x = self.fc(x_img)  # (B, N, R)

        else:
            x = self.fc(x)  # (B, N, R)

        if self.use_one_image:
            x = x[:, self.n_seq_before, :].unsqueeze(1)

        return x

class LayerFcX(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.H = config.fc_hidden_size
        self.fc = nn.Sequential(
            nn.Linear(self.H, self.H),
        )

    def forward(self, x):  # x: (B, N, H)
        return self.fc(x)  # (B, N, H)


class LayerBiLstmX(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.H = config.fc_hidden_size
        self.lstm = nn.LSTM(input_size=self.H,
                            hidden_size=self.H // 2,
                            num_layers=config.lstm_layer_num,
                            bidirectional=True,
                            batch_first=True)

    def forward(self, x):  # x: (B, N, H)
        return self.lstm(x, hx=None)[0]  # (B, N, H)


class LayerSelfAttnX(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.H = config.fc_hidden_size

        # key layer
        self.attn = nn.MultiheadAttention(self.H, num_heads=1, dropout=0.2, batch_first=True)

        # attention map
        self.save_attention = config.enable_save_attention
        self.attn_map = None
        self.attn_out = None
        self.final_out = None

    def forward(self, x):  # x: (B, N, H)
        if self.save_attention:
            self.attn_out, self.attn_map = self.attn(x, x, x)  # (B, N, H), (B, N, N)
            self.final_out = self.attn_out
            return self.final_out
        else:
            return self.attn(x, x, x)[0]


class LayerFinal(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.N = 1 if config.model.startswith('CF1') else config.n_seq_total
        self.H = config.fc_hidden_size

        self.enable_layer_norm = config.enable_layer_norm
        self.layer_norm = nn.LayerNorm(self.H) if config.enable_layer_norm else nn.Identity()

        self.enable_dropout = config.enable_dropout
        self.dropout = nn.Dropout(config.dropout) if config.enable_dropout else nn.Identity()

        self.relu = nn.ReLU()

        self.fc = nn.Linear(self.N * self.H, config.output_size)

        self.save_activation = config.enable_save_activation
        self.activation = None  # record if each neuron is activated

    def forward(self, x):  # x: (B, N, H)
        if self.enable_layer_norm:
            x = self.layer_norm(x)

        x = x.reshape(x.size(0), -1)

        if self.enable_dropout:
            x = self.dropout(x)

        x = self.relu(x)
        if self.save_activation:
            self.activation = x

        return self.fc(x)  # (B, output_size)


class OurModel(nn.Module):
    def __init__(self, config, st_layer):
        super().__init__()

        # image processing Layer
        self.conv = LayerImageConvolution(config)

        # ST (spatio-temporal) layer
        self.st_layer = st_layer

        self.enable_residual_gamma = config.enable_residual_gamma
        # if config.model.endswith('Null'):
        #     self.enable_residual_gamma = False
        if self.enable_residual_gamma:
            self.gamma = nn.Parameter(torch.zeros(1))

        # final layer
        self.final = LayerFinal(config)

    def forward(self, x):
        x = self.conv(x)  # (B, N, R)

        if self.enable_residual_gamma:
            x = self.gamma * self.st_layer(x) + x
        else:
            x = self.st_layer(x)  # (B, N, H)

        x = self.final(x)  # (B, output_size)
        return x


def get_model(config):
    model_name = config.model
    if model_name == 'CF1Null' or model_name == 'CF2Null':
        model = OurModel(config, nn.Identity())
    elif model_name == 'CF1X' or model_name == 'CF2X':
        model = OurModel(config, LayerFcX(config))
    elif model_name == "CBLX":
        model = OurModel(config, LayerBiLstmX(config))
    elif model_name == "CSAX":
        model = OurModel(config, LayerSelfAttnX(config))
    else:
        raise RuntimeError(f"invalid model name {model_name}")

    model = model.to(config.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.lr,
                                 weight_decay=config.wd if config.enable_weight_decay else 0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                       gamma=config.lr_gamma)
    return model, criterion, optimizer, scheduler


if __name__ == '__main__':
    config = load_config()

    ''' Design input '''
    B = config.batch_size  # batch size
    C = config.img_colors
    H = config.img_size
    W = config.img_size
    N = config.n_seq_total  # sequence length
    R = config.resnet_fc_size  # resnet output size from its fc layer

    if config.enable_resnet_in_model:
        x = torch.randn(B, N, C, H, W)
    else:
        x = torch.randn(B, N, R)

    x = x.to(config.device)
    print("input shape:", x.shape)
    print("input device:", x.device)


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
    model_names = ['CF1Null', 'CF2Null', 'CF1X', 'CF2X', 'CSAX', 'CBLX', ]
    total_params_list_list = []
    stl_params_list_list = []
    stl_params_pct_list = []
    H_list = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    # H_list = [20, 40, 50, 80, 120, 160, 200]
    for H in H_list:
        total_params_list = []
        stl_params_list = []
        stl_params_pct = []
        config.fc_hidden_size = H
        print(f"\nconfig.fc_hidden_size = {H}")
        for model_name in model_names:
            print('\n')
            print('>' * 50)
            print('model: ', model_name)
            config.model = model_name
            model, _, _, _ = get_model(config)
            total_params = get_total_parameter_num(model)
            stl_params = get_total_parameter_num(model.st_layer) + 1 if config.enable_residual_gamma and not model_name.endswith("Null") else 0
            stl_pct = stl_params / total_params * 100
            total_params_list.append(total_params)
            stl_params_list.append(stl_params)
            stl_params_pct.append(stl_pct)
            print('model param size: ', total_params)
            print('STL param size: ', stl_params)
            print('STL param pct: ', stl_pct)
            print('model neuron num: ', get_total_neuron_num(model))
            y = model(x)
            print("output shape", y.shape)

            # from torchinfo import summary
            # summary(model, input_size=x.shape)

        total_params_list_list.append(total_params_list)
        stl_params_list_list.append(stl_params_list)
        stl_params_pct_list.append(stl_params_pct)

    print("H_list", H_list)
    print("Param list\n", total_params_list_list)

    '''Create the DataFrame for printing '''
    import pandas as pd
    data = {}
    for idx, model_name in enumerate(model_names):
        for H_idx, H in enumerate(H_list):
            col_name = f"H={H}, {model_name}"
            data[col_name] = [
                total_params_list_list[H_idx][idx],
                stl_params_list_list[H_idx][idx],
                stl_params_pct_list[H_idx][idx]
            ]

    df = pd.DataFrame(data, index=['Total Params Size', 'STL Params Size', 'STL Params Pct'])

    # Transpose the DataFrame to match the desired structure
    df = df.transpose()

    # Reset the index to add columns H and model_name
    df.reset_index(inplace=True)
    df[['H', 'model_name']] = df['index'].str.split(',', expand=True)
    df.drop(columns=['index'], inplace=True)

    # Reorder the columns
    df = df[[ 'model_name', 'H', 'Total Params Size', 'STL Params Size', 'STL Params Pct']]

    print(df)
