import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter
import torch.nn.init as init
import math

class Linear(Module):
    """Aplica uma transformação linear aos dados de entrada."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Inicializa os pesos usando Kaiming Uniform."""
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Executa a multiplicação linear."""
        if input.shape[-1] != self.in_features:
            raise RuntimeError(f"Dimensão de entrada inválida: esperado {self.in_features}, mas recebeu {input.shape[-1]}")
        return F.linear(input, self.weight, self.bias)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Executa a multiplicação linear."""
        if input.shape[-1] != self.in_features:
            raise RuntimeError(f"Dimensão de entrada inválida: esperado {self.in_features}, mas recebeu {input.shape[-1]}")

        # Implementação correta da multiplicação linear
        return F.linear(input, self.weight, self.bias)
    
    def reset_parameters(self) -> None:
        """Inicializa os pesos usando Kaiming Uniform."""
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)


