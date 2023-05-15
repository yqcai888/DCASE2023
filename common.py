import torch
import torch.nn as nn


class ShuffleLayer(nn.Module):
    def __init__(self, group=10):
        super(ShuffleLayer, self).__init__()
        self.group = group

    def forward(self, x):
        b, c, f, t = x.data.size()
        assert c % self.group == 0
        group_channels = c // self.group

        x = x.reshape(b, group_channels, self.group, f, t)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, c, f, t)
        return x


class AdaResNorm(nn.Module):
    def __init__(self, c, grad=False, id_norm=None, eps=1e-5):
        super(AdaResNorm, self).__init__()
        self.grad = grad
        self.id_norm = id_norm
        self.eps = torch.Tensor(1, c, 1, 1)
        self.eps.data.fill_(eps)

        if self.grad:
            self.rho = nn.Parameter(torch.Tensor(1, c, 1, 1))
            self.rho.data.fill_(0.5)
            self.gamma = nn.Parameter(torch.ones(1, c, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        else:
            self.rho = torch.Tensor(1, c, 1, 1)
            self.rho.data.fill_(0.5)

    def forward(self, x):
        self.eps = self.eps.to(x.device)
        self.rho = self.rho.to(x.device)

        identity = x
        ifn_mean = x.mean((1, 3), keepdim=True)
        ifn_var = x.var((1, 3), keepdim=True)
        ifn = (x - ifn_mean) / (ifn_var + self.eps).sqrt()

        res_norm = self.rho * identity + (1 - self.rho) * ifn

        if self.grad:
            return self.gamma * res_norm + self.beta
        else:
            return res_norm


class ConvBlock(nn.Module):
    """
    Standard convolution block with Batch normalization and activation.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 use_bn=False,
                 use_relu=False):
        super(ConvBlock, self).__init__()
        self.use_bn = use_bn
        self.use_relu = use_relu

        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if self.use_relu:
            self.activation = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_relu:
            x = self.activation(x)
        return x


class TimeFreqSeparableConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout_rate=0.2,
                 shuffle=False,
                 shuffle_groups=10):
        super(TimeFreqSeparableConv, self).__init__()
        self.transition = in_channels != out_channels
        self.shuffle = shuffle
        self.half_channels = out_channels // 2

        if self.transition:
            self.trans_conv = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                        use_bn=True, use_relu=True)
        self.freq_dw_conv = ConvBlock(in_channels=self.half_channels, out_channels=self.half_channels,
                                      kernel_size=(3, 1),
                                      padding=(1, 0), groups=self.half_channels, use_bn=True, use_relu=True)
        self.temp_dw_conv = ConvBlock(in_channels=self.half_channels, out_channels=self.half_channels,
                                      kernel_size=(1, 3),
                                      padding=(0, 1), groups=self.half_channels, use_bn=True, use_relu=True)
        self.freq_pw_conv = ConvBlock(in_channels=self.half_channels, out_channels=self.half_channels, kernel_size=1,
                                      use_bn=True, use_relu=True)
        self.temp_pw_conv = ConvBlock(in_channels=self.half_channels, out_channels=self.half_channels, kernel_size=1,
                                      use_bn=True, use_relu=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.shuffle_layer = ShuffleLayer(group=shuffle_groups)

    def forward(self, x):
        # Expand or shrink channels if in_channels != out_channels
        if self.transition:
            x = self.trans_conv(x)
        # Channel shuffle
        if self.shuffle:
            x = self.shuffle_layer(x)
        # Split feature maps into two halves on the channel dimension
        x1, x2 = torch.split(x, self.half_channels, dim=1)
        # Copy x1, x2 for residual path
        identity1 = x1
        identity2 = x2
        # Frequency-wise convolution block
        x1 = self.freq_dw_conv(x1)
        x1 = x1.mean(2, keepdim=True)  # frequency average pooling
        x1 = self.freq_pw_conv(x1)
        x1 = self.dropout(x1)
        x1 = x1 + identity1
        # Time-wise convolution block
        x2 = self.temp_dw_conv(x2)
        x2 = x2.mean(3, keepdim=True)  # temporal average pooling
        x2 = self.temp_pw_conv(x2)
        x2 = self.dropout(x2)
        x2 = x2 + identity2
        # Concat x1 and x2
        x = torch.cat((x1, x2), dim=1)
        return x