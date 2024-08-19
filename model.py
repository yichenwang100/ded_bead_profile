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
        self.mae_loss = nn.L1Loss()
        self.mae_loss_lambda = config.criterion_mae_lambda

        self.mse_loss = nn.MSELoss()
        self.mse_loss_lambda = config.criterion_mse_lambda

        self.iou_loss = MyIoULoss(config)
        self.iou_loss_lambda = config.criterion_iou_lambda

    def forward(self, y_pred, y_true):
        mae_loss = self.mae_loss(y_pred, y_true)
        mse_loss = self.mse_loss(y_pred, y_true)
        iou_loss = self.iou_loss(y_pred, y_true)
        return (mae_loss * self.mae_loss_lambda +
                mse_loss * self.mse_loss_lambda +
                iou_loss * self.iou_loss_lambda)


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

        embed_dim = (config.img_embed_dim +
                     config.param_embed_dim * config.param_size +
                     config.pos_embed_dim * config.pos_size)
        if embed_dim != config.embed_dim:
            raise RuntimeError(f"calculated embed_dim({embed_dim}) != config.embed_dim({config.embed_dim})")

        self.param_embed = MyEmbeddingBlock(config.param_size, config.param_embed_dim, config.feature_embed_option)
        self.pos_embed = MyEmbeddingBlock(config.pos_size, config.pos_embed_dim, config.feature_embed_option)

    def forward(self, x_img, x_param, x_pos):
        x_param = self.param_embed(x_param)
        x_pos = self.pos_embed(x_pos)
        x = torch.cat((x_img, x_param, x_pos), dim=2)
        return x  # (B, N, H_b)


class MyFeatureAttnBlock(nn.Module):
    def __init__(self, config, *args, **kwargs):
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
    def __init__(self, config, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # temporal wise attn
        self.lstm = nn.LSTM(input_size=self.hidden_dim,
                            hidden_size=self.hidden_dim // 2,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True,
                            )

        # layer norm
        self.enable_layer_norm = config.enable_layer_norm
        if self.enable_layer_norm:
            self.ln = nn.LayerNorm(self.hidden_dim)

    def forward(self, x):
        x = self.lstm(x)[0]

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
    def __init__(self, config):
        super().__init__()

        layers = []
        if config.model == 'STEN_GP_FFD_BLSTM':
            for _ in range(config.encoder_layer_size):
                layers.append(MyFeedForwardBlock(config, config.embed_dim))
                layers.append(MyBiLSTMBlock(config, config.embed_dim))
        elif config.model == 'STEN_GP_FFD_TA':
            for _ in range(config.encoder_layer_size):
                layers.append(MyFeedForwardBlock(config, config.embed_dim))
                layers.append(MyTemporalAttnBlock(config))
        elif config.model == 'STEN_GP_TA_FFD':
            for _ in range(config.encoder_layer_size):
                layers.append(MyTemporalAttnBlock(config))
                layers.append(MyFeedForwardBlock(config, config.embed_dim))
        elif config.model == 'STEN_GP_FA_TA':
            for _ in range(config.encoder_layer_size):
                layers.append(MyFeatureAttnBlock(config))
                layers.append(MyTemporalAttnBlock(config))
        elif config.model == 'STEN_GP_FFD':
            for _ in range(config.encoder_layer_size):
                layers.append(MyFeedForwardBlock(config, config.embed_dim))
        else:  # default setup
            for _ in range(config.encoder_layer_size):
                layers.append(MyFeatureAttnBlock(config))
                layers.append(MyTemporalAttnBlock(config))
                layers.append(MyFeedForwardBlock(config, config.embed_dim))

        self.encoder = nn.Sequential(*layers)

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
        self.st_layer = nn.Sequential(
            MyFeedForwardBlock(config, self.output_latent_dim),
            MyBiLSTMBlock(config, self.output_latent_dim)
        )

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

    def forward(self, x_latent, y_prev):
        y_embed = self.embed(y_prev)
        y_latent = self.st_layer(y_embed)[:, -1, :]

        x_latent = x_latent[:, self.n_seq_before, :]
        x = torch.cat((x_latent, y_latent), dim=-1)
        x = self.final(x)
        return x


class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = MyInputEmbedding(config)
        self.encoder = MyEncoder(config)
        if 'decoder_option' in config and config.decoder_option == 'transformer':
            self.decoder = MyDecoderPlus(config)
        else:
            self.decoder = MyDecoder(config)
        self.timing_on = False

    def forward(self, x_img, x_param, x_pos, y_prev):
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
                x = self.decoder(x_enc, y_prev)  # (B, N, output_size)
            time_dec = time.time()
            print(f'> model timing: ')
            print(f'time_embed_elapsed, {(time_embed - time_start) / time_count * 1e6:.3f}us')
            print(f'time_enc_elapsed, {(time_enc - time_embed) / time_count * 1e6:.3f}us')
            print(f'time_dec_elapsed, {(time_dec - time_enc) / time_count * 1e6:.3f}us')
        else:
            x = self.embedding(x_img, x_param, x_pos)  # (B, N, H_b)
            x = self.encoder(x)  # (B, N, H_c)
            x = self.decoder(x, y_prev)  # (B, N, output_size)
        return x


def get_model(config):
    model = MyModel(config).to(config.device)

    # auto parallel in multiple gpu
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if 'enable_param_init' in config and config.enable_param_init:
        model.apply(init_model_weights)

    # criterion
    if 'criterion_option' in config and config.criterion_option == 'iou':
        criterion = MyIoULoss(config)
    elif 'criterion_option' in config and config.criterion_option == 'mse_iou':
        criterion = MyCombinedLoss(config)
    elif 'criterion_option' in config and config.criterion_option == 'mae_mse_iou':
        criterion = MyCombinedLoss(config)
    else:
        criterion = nn.MSELoss()

    # optimizer
    if config.optimizer_option == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=config.lr,
                                      weight_decay=config.wd if config.enable_weight_decay else 0)
    elif config.optimizer_option == 'adam_Vaswani_2017':
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
    ENABLE_PROFILING = False

    config = load_config()


    ''' I/O test '''
    # model_names = ['STEN_GP_FFD_TA', 'STEN_GP_FFD_BLSTM']
    model_names = ['STEN_GP_FFD_BLSTM']
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

        print(f"\n> model design...")
        for model_name in model_names:
            print('\n')
            print('>' * 50)
            print('> model: ', model_name)
            config.model = model_name
            model, _, _, _ = get_model(config)
            total_params = get_model_parameter_num(model)
            total_params_list.append(total_params)
            print('> model param size: ', total_params)
            print('> model neuron num: ', get_model_neuron_num(model))
            model.timing_on = False  # print timing of each layer
            if config.decoder_option == 'transformer':
                y = model(x_img, x_param, x_pos, y_prev)
            else:
                y = model(x_img, x_param, x_pos)
            print("> output shape", y.shape)

            from torchinfo import summary

            model.timing_on = False
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
                        output = model.embedding(x_img, x_param, x_pos)
                        prof.step()

                print(f"total elapsed time: {(time.time() - time_start) * 1000:.3f}ms")
                print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

        total_params_list_list.append(total_params_list)


    for i_list, _param_list in enumerate(total_params_list_list):
        print(f"> Param list for H = {H_list[i_list]}: ",
              [int_split_by_comma(num) for num in _param_list])
