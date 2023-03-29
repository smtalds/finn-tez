from torch import nn
import brevitas.nn as qnn


class AI85KWS20Netv3_brevitas(nn.Module):
    """
    Compound KWS20 v3 Audio net, all with Conv1Ds, but with brevitas
    """

    # num_classes = n keywords + 1 unknown
    def __init__(
            self,
            num_classes=21,
            num_channels=128,
            dimensions=(128, 1),  # pylint: disable=unused-argument
            bias=False,
            weight_quant_type='INT',
            weight_bit_width=4,
            weight_scaling_impl_type='CONST',
            weight_scaling_const=1,
            bias_quant_type='INT',
            bias_bit_width=4,
            act_quant_type='INT',
            act_bit_width=4,
            act_scaling_impl_type='CONST',
            act_scaling_const=1,
            **kwargs):
        super().__init__()
        self.indentity = qnn.QuantIdentity(bit_width=4,
                                           return_quant_tensor=False)
        self.voice_conv1 = qnn.QuantConv1d(num_channels, 100, 1, bias=False,
                                           stride=1, padding=0,
                                           weight_bit_width=4,
                                           )
        self.voice_conv1_relu = qnn.QuantReLU(bit_width=4,
                                              return_quant_tensor=False)

        # T: 128 F: 100
        self.voice_conv2 = qnn.QuantConv1d(100, 96, 3, bias=False, stride=1,
                                           padding=0, weight_bit_width=4,
                                           )
        self.voice_conv2_relu = qnn.QuantReLU(bit_width=4,
                                              return_quant_tensor=False)
        self.dropout1 = qnn.QuantDropout(p=0.5)
        # T: 126 F : 96
        self.voice_conv3 = qnn.QuantConv1d(96, 64, 3, bias=False, stride=1,
                                           padding=0, weight_bit_width=4,
                                           )
        self.voice_conv3_maxpool = qnn.QuantMaxPool1d(kernel_size=2, stride=2)
        self.voice_conv3_relu = qnn.QuantReLU(bit_width=4,
                                              return_quant_tensor=False)
        # T: 62 F : 64
        self.voice_conv4 = qnn.QuantConv1d(64, 48, 3, bias=False, stride=1,
                                           padding=0, weight_bit_width=4,
                                           )
        self.voice_conv4_relu = qnn.QuantReLU(bit_width=4,
                                              return_quant_tensor=False)
        self.dropout2 = qnn.QuantDropout(p=0.5)
        # T : 60 F : 48
        self.kws_conv1 = qnn.QuantConv1d(48, 64, 3, bias=False, stride=1,
                                         padding=0, weight_bit_width=4,
                                         )
        self.kws_conv1_maxpool = qnn.QuantMaxPool1d(kernel_size=2, stride=2)
        self.kws_conv1_relu = qnn.QuantReLU(bit_width=4,
                                            return_quant_tensor=False)

        # T: 30 F : 64
        self.kws_conv2 = qnn.QuantConv1d(64, 96, 3, bias=False,
                                         stride=1, padding=0,
                                         weight_bit_width=4,
                                         )
        self.kws_conv2_relu = qnn.QuantReLU(bit_width=4,
                                            return_quant_tensor=False)
        self.dropout3 = qnn.QuantDropout(p=0.5)
        # T: 28 F : 96
        self.kws_conv3 = qnn.QuantConv1d(96, 100, 3, bias=False,
                                         stride=1, padding=0,
                                         weight_bit_width=4,
                                         )
        self.kws_conv3_relu = qnn.QuantReLU(bit_width=4,
                                            return_quant_tensor=False)

        # T : 14 F: 100
        self.kws_conv4 = qnn.QuantConv1d(100, 64, 6, bias=False,
                                         stride=1, padding=0,
                                         weight_bit_width=4)
        self.kws_conv4_maxpool = qnn.QuantMaxPool1d(kernel_size=2, stride=2)
        self.kws_conv4_relu = qnn.QuantReLU(bit_width=4,
                                            return_quant_tensor=False)
        # T : 2 F: 128
        self.fc = nn.Linear(640, num_classes, bias=False)
        
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        # Run CNN
        x = self.indentity(x)
        x = self.voice_conv1(x)
        x = self.voice_conv1_relu(x)
        x = self.voice_conv2(x)
        x = self.voice_conv2_relu(x)
        x = self.dropout1(x)
        x = self.voice_conv3(x)
        x = self.voice_conv3_maxpool(x)
        x = self.voice_conv3_relu(x)
        x = self.voice_conv4(x)
        x = self.voice_conv4_relu(x)
        x = self.dropout2(x)
        x = self.kws_conv1(x)
        x = self.kws_conv1_maxpool(x)
        x = self.kws_conv1_relu(x)
        x = self.kws_conv2(x)
        x = self.dropout3(x)
        x = self.kws_conv2_relu(x)
        x = self.kws_conv3(x)
        x = self.kws_conv3_relu(x)
        x = self.kws_conv4(x)
        x = self.kws_conv4_maxpool(x)
        x = self.kws_conv4_relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def samet_brevitas(pretrained=False, **kwargs):
    """
    Constructs a AI85KWS20Net model.
    rn AI85KWS20Net(**kwargs)
    """
    assert not pretrained
    return AI85KWS20Netv3_brevitas(**kwargs)


models = [
    {
        'name': 'samet_brevitas',
        'min_input': 1,
        'dim': 1,
    },
]
