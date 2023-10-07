import torch.nn as nn
import torch.nn.functional as F
from brainbox.models import BBModel


class LayerModel(BBModel):

    def __init__(self, n_in, n_out, rf_len, rf_size, stride, recurrent=False):
        super().__init__()
        self._n_in = n_in
        self._n_out = n_out
        self._rf_len = rf_len
        self._rf_size = rf_size
        self._stride = stride
        self._recurrent = recurrent

        self.encoder = nn.Conv3d(n_in, n_out, (rf_len, rf_size, rf_size), (1, stride, stride))
        self.decoder = nn.ConvTranspose3d(n_out, n_in, (1, rf_size, rf_size), (1, stride, stride))
        self.init_weight(self.encoder.weight, "glorot_normal")
        self.init_weight(self.decoder.weight, "glorot_normal")

    @property
    def hyperparams(self):
        return {**super().hyperparams, "n_in": self._n_in, "n_out": self._n_out, "rf_len": self._rf_len, "rf_size": self._rf_size, "stride": self._stride, "recurrent": self._recurrent}

    def forward(self, x):
        hidden = self.encoder(x)
        hidden = F.softplus(hidden, 10, 20)
        output = self.decoder(hidden)

        return hidden, output


class StackModel(BBModel):

    def __init__(self, recurrent=False):
        super().__init__()
        self._recurrent = recurrent

        self._layer1 = LayerModel(n_in=1, n_out=100, rf_len=5, rf_size=20, stride=4, recurrent=recurrent)
        self._layer2 = LayerModel(n_in=100, n_out=100, rf_len=5, rf_size=3, stride=1, recurrent=recurrent)
        self._layer3 = LayerModel(n_in=100, n_out=200, rf_len=5, rf_size=3, stride=1, recurrent=recurrent)
        self._layer4 = LayerModel(n_in=200, n_out=400, rf_len=5, rf_size=3, stride=1, recurrent=recurrent)

    @property
    def hyperparams(self):
        return {**super().hyperparams, "recurrent": self._recurrent}

    def get_params(self):
        weights = [layer.encoder.weight for layer in [self._layer1, self._layer2, self._layer3, self._layer4]]
        weights.extend([layer.decoder.weight for layer in [self._layer1, self._layer2, self._layer3, self._layer4]])

        return weights

    def forward(self, x, layer=None):
        # x: b x n x t x h x w

        h0 = x
        h1, o1 = self._layer1(h0)
        if layer == 1:
            return h1
        h2, o2 = self._layer2(h1)
        if layer == 2:
            return h2
        h3, o3 = self._layer3(h2)
        if layer == 3:
            return h3
        h4, o4 = self._layer4(h3)
        if layer == 4:
            return h4

        return [(o1, h0, h1), (o2, h1, h2), (o3, h2, h3), (o4, h3, h4)]
