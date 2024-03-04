from torch import nn, Tensor


class SegModel(nn.Module):
    def __init__(
            self,
            encoder: nn.Module,
            decoder: nn.Module
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, image: Tensor) -> Tensor:
        features = self.encoder(image)
        return self.decoder(features)

    def get_encoder(self) -> nn.Module:
        return self.encoder

    def load_weights(self, model: nn.Module) -> None:
        self.load_state_dict(model.state_dict())
