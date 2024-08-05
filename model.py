import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from util import *


class MyIoULoss(nn.Module):
    def __init__(self, config, smooth=1e-6):
        super().__init__()
        # self.smooth = smooth
        self.noise_cutoff = config.output_noise_cutoff

    def forward(self, y_pred, y_true):
        # preds = preds.view(-1)
        # targets = targets.view(-1)
        # intersection = (preds * targets).sum()
        # union = preds.sum() + targets.sum() - intersection
        # iou = (intersection + self.smooth) / (union + self.smooth)

        iou = compute_iou(y_pred, y_true, noise_cutoff=self.noise_cutoff).mean()
        return 1 - iou


class MyCombinedLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.mse_loss_lambda = config.criterion_mse_lambda
        self.iou_loss = MyIoULoss(config)
        self.iou_loss_lambda = config.criterion_iou_lambda

    def forward(self, y_pred, y_true):
        mse_loss = self.mse_loss(y_pred, y_true)
        iou_loss = self.iou_loss(y_pred, y_true)
        return (mse_loss * self.mse_loss_lambda + iou_loss * self.iou_loss_lambda)


class MyEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        N = config.n_seq_total
        self.H_param = config.param_size
        self.H_pos = config.pos_size

        self.enable_linear = False
        if 'feature_embed_option' in config and config.feature_embed_option == 'fc':
            self.param_weight = nn.Parameter(torch.randn(self.H_param, config.param_embed_dim))
            self.param_bias = nn.Parameter(torch.randn(1, config.param_embed_dim))
            self.pos_weight = nn.Parameter(torch.randn(self.H_pos, config.pos_embed_dim))
            self.param_bias = nn.Parameter(torch.randn(1, config.pos_embed_dim))
            self.enable_linear = True
        else:
            self.param_embed = nn.ModuleList([nn.Linear(1, config.param_embed_dim) for _ in range(self.H_param)])
            self.pos_embed = nn.ModuleList([nn.Linear(1, config.pos_embed_dim) for _ in range(self.H_pos)])
            self.embed_dim = (config.img_embed_dim +
                              config.param_embed_dim * config.param_size +
                              config.pos_embed_dim * config.pos_size)
            if self.embed_dim != config.embed_dim:
                raise RuntimeError(f"self.embed_dim({self.embed_dim}) != config.embed_dim({config.embed_dim})")

    def forward(self, x_img, x_param, x_pos):
        if self.enable_linear:
            x_param = x_param.expand(self.H_param, -1, -1, -1).permute(1, 2, 3, 0)
            x_param = x_param.matmul(self.param_weight) + self.param_bias
            x_param = x_param.view(x_img.size(0), x_img.size(1), -1)
            x_pos = x_pos.expand(self.H_pos, -1, -1, -1).permute(1, 2, 3, 0)
            x_pos = x_pos.matmul(self.pos_weight) + self.param_bias
            x_pos = x_pos.view(x_img.size(0), x_img.size(1), -1)
        else:
            x_param = torch.cat([self.param_embed[i](x_param[:, :, i].unsqueeze(-1)) for i in range(self.H_param)],
                                dim=-1)
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

        # save attention
        self.enable_save_attention = config.enable_save_attention
        if self.enable_save_attention:
            self.attn_map = None

        # layer norm
        self.enable_layer_norm = config.enable_layer_norm
        if self.enable_layer_norm:
            self.ln = nn.LayerNorm(config.n_seq_total)

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


class MyBiLSTMBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        # temporal wise attn
        self.lstm = nn.LSTM(input_size=config.embed_dim,
                            hidden_size=config.embed_dim // 2,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True,
                            )

        # layer norm
        self.enable_layer_norm = config.enable_layer_norm
        if self.enable_layer_norm:
            self.ln = nn.LayerNorm(config.embed_dim)

    def forward(self, x):
        x = self.lstm(x)[0]

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
            x = x + self.fc2(self.relu(self.fc1(x)))

        if self.enable_layer_norm:
            x = self.ln(x)
        return x


class MyEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        layers = []
        if config.model == 'STEN_GP_FFD_BLSTM':
            for _ in range(config.encoder_layer_size):
                layers.append(MyFeedForwardBlock(config))
                layers.append(MyBiLSTMBlock(config))
        elif config.model == 'STEN_GP_FFD_TA':
            for _ in range(config.encoder_layer_size):
                layers.append(MyFeedForwardBlock(config))
                layers.append(MyTemporalAttnBlock(config))
        elif config.model == 'STEN_GP_TA_FFD':
            for _ in range(config.encoder_layer_size):
                layers.append(MyTemporalAttnBlock(config))
                layers.append(MyFeedForwardBlock(config))
        elif config.model == 'STEN_GP_FA_TA':
            for _ in range(config.encoder_layer_size):
                layers.append(MyFeatureAttnBlock(config))
                layers.append(MyTemporalAttnBlock(config))
        elif config.model == 'STEN_GP_FFD':
            for _ in range(config.encoder_layer_size):
                layers.append(MyFeedForwardBlock(config))
        else:  # default setup
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


class OurModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = MyEmbedding(config)
        self.encoder = MyEncoder(config)
        self.decoder = MyDecoder(config)
        self.timing_on = False

    def forward(self, x_img, x_param, x_pos):
        if self.timing_on:
            time_count = 100

            time_start = time.time()
            for i in range(time_count):
                x_embed = self.embedding(x_img, x_param, x_pos)  # (B, N, H_b)
            time_embed = time.time()

            for i in range(time_count):
                x_enc = self.encoder(x_embed)  # (B, N, H_c)
            time_enc = time.time()

            for i in range(time_count):
                x = self.decoder(x_enc)  # (B, N, output_size)
            time_dec = time.time()
            print(f'> model timing: ')
            print(f'time_embed_elapsed, {(time_embed - time_start) / time_count * 1e6:.3f}us')
            print(f'time_enc_elapsed, {(time_enc - time_embed) / time_count * 1e6:.3f}us')
            print(f'time_dec_elapsed, {(time_dec - time_enc) / time_count * 1e6:.3f}us')
        else:
            x = self.embedding(x_img, x_param, x_pos)  # (B, N, H_b)
            x = self.encoder(x)  # (B, N, H_c)
            x = self.decoder(x)  # (B, N, output_size)
        return x


def get_model(config):
    model = OurModel(config).to(config.device)

    # criterion
    if 'criterion_option' in config and config.criterion_option == 'iou':
        criterion = MyIoULoss(config)
    elif 'criterion_option' in config and config.criterion_option == 'mse_iou':
        criterion = MyCombinedLoss(config)
    else:
        criterion = nn.MSELoss()

    # optimizer
    if config.optimizer_option == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=config.lr,
                                      weight_decay=config.wd if config.enable_weight_decay else 0)
    elif config.optimizer_option == 'adam_opt1':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=config.lr,
                                      betas=(0.9, 0.98),
                                      weight_decay=config.wd if config.enable_weight_decay else 0)
    else:
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
    model_names = ['STEN_GP_FFD_TA', 'STEN_GP_FFD_BLSTM']
    total_params_list_list = []
    # H_list = [4, 6, 8]
    H_list = [6]
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
            model.timing_on = True  # print timing of each layer
            y = model(x_img, x_param, x_pos)
            print("> output shape", y.shape)

            from torchinfo import summary

            model.timing_on = False
            summary(model, input_size=[x_img.shape, x_param.shape, x_pos.shape])

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
                    output = model.embedding(x_img, x_param, x_pos)
                    prof.step()

            print(f"total elapsed time: {(time.time() - time_start) * 1000:.3f}ms")
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

        total_params_list_list.append(total_params_list)

    import re


    def format_number(number):
        return re.sub(r'(?<!^)(?=(\d{3})+$)', ',', str(number))


    for i_list, _param_list in enumerate(total_params_list_list):
        print(f"> Param list for H = {H_list[i_list]}: ",
              [format_number(num) for num in _param_list])
