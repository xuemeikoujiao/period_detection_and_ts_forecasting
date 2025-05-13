# # Copyright © 2020 Element AI Inc. All rights reserved.
# """
# N-BEATS Model.
# """
from typing import Tuple

import numpy as np
import torch as t

class GenericBasis(t.nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: t.Tensor):
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]

class TrendBasis(t.nn.Module):
    def __init__(self, degree_of_polynomial: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.polynomial_size = degree_of_polynomial + 1
        self.backcast_time = t.nn.Parameter(
            t.tensor(np.concatenate([np.power(np.arange(backcast_size, dtype=float) / backcast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=t.float32),
            requires_grad=False)
        self.forecast_time = t.nn.Parameter(
            t.tensor(np.concatenate([np.power(np.arange(forecast_size, dtype=float) / forecast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=t.float32), requires_grad=False)

    def forward(self, theta: t.Tensor):
        backcast = t.einsum('bp,pt->bt', theta[:, self.polynomial_size:], self.backcast_time)
        forecast = t.einsum('bp,pt->bt', theta[:, :self.polynomial_size], self.forecast_time)
        return backcast, forecast



class SeasonalityBasis(t.nn.Module):
    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.frequency = t.arange(1, harmonics + 1, dtype=t.float32)[None, :]
        forecast_scale = 1 / forecast_size
        backcast_grid = -2 * np.pi * (t.arange(backcast_size, dtype=t.float32)[:, None] * forecast_scale) * self.frequency
        forecast_grid = 2 * np.pi * (t.arange(forecast_size, dtype=t.float32)[:, None] * forecast_scale) * self.frequency
        self.backcast_cos_template = t.nn.Parameter(t.cos(backcast_grid).T, requires_grad=False)
        self.backcast_sin_template = t.nn.Parameter(t.sin(backcast_grid).T, requires_grad=False)
        self.forecast_cos_template = t.nn.Parameter(t.cos(forecast_grid).T, requires_grad=False)
        self.forecast_sin_template = t.nn.Parameter(t.sin(forecast_grid).T, requires_grad=False)

    def forward(self, theta: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        params_per_harmonic = self.backcast_cos_template.shape[0]
        backcast_harmonics_cos = t.einsum('bp,pt->bt', theta[:, 2 * params_per_harmonic:3 * params_per_harmonic],
                                          self.backcast_cos_template)
        backcast_harmonics_sin = t.einsum('bp,pt->bt', theta[:, 3 * params_per_harmonic:], self.backcast_sin_template)
        backcast = backcast_harmonics_cos + backcast_harmonics_sin
        forecast_harmonics_cos = t.einsum('bp,pt->bt', theta[:, :params_per_harmonic], self.forecast_cos_template)
        forecast_harmonics_sin = t.einsum('bp,pt->bt', theta[:, params_per_harmonic:2 * params_per_harmonic],
                                          self.forecast_sin_template)
        forecast = forecast_harmonics_cos + forecast_harmonics_sin
        return backcast, forecast
    

class NBeatsBlock(t.nn.Module):
    def __init__(self, input_size, theta_size: int, basis_function: t.nn.Module, layers: int, layer_size: int):
        super().__init__()
        self.layers = t.nn.ModuleList([t.nn.Linear(input_size, layer_size)] +
                                      [t.nn.Linear(layer_size, layer_size) for _ in range(layers - 1)])
        self.basis_parameters = t.nn.Linear(layer_size, theta_size)
        self.basis_function = basis_function

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        block_input = x
        for layer in self.layers:
            block_input = t.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)

class NBeats(t.nn.Module):
    def __init__(self, blocks: t.nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    # def forward(self, x: t.Tensor, input_mask: t.Tensor) -> t.Tensor:
    #     residuals = x.flip(dims=(1,))
    #     input_mask = input_mask.flip(dims=(1,))
    #     forecast = x[:, -1:]
    #     for block in self.blocks:
    #         backcast, block_forecast = block(residuals)
    #         residuals = (residuals - backcast) * input_mask
    #         forecast = forecast + block_forecast
    #     return forecast
    def forward(self, x: t.Tensor, input_mask: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        residuals = x.flip(dims=(1,))
        input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]
        backcast = t.zeros_like(residuals)  # 初始化 backcast
        for block in self.blocks:
            block_backcast, block_forecast = block(residuals)
            residuals = (residuals - block_backcast) * input_mask
            forecast = forecast + block_forecast
            backcast = backcast + block_backcast  # 累积 backcast
        return backcast, forecast

def create_model(device, input_size=100, forecast_size=7):
    blocks = t.nn.ModuleList()
    # Generic block
    blocks.append(NBeatsBlock(
        input_size=input_size,
        theta_size=input_size + forecast_size,
        basis_function=GenericBasis(input_size, forecast_size),
        layers=4,
        layer_size=512
    ))
    # Trend block
    blocks.append(NBeatsBlock(
        input_size=input_size,
        theta_size=2*(3+1),  # poly degree 3
        basis_function=TrendBasis(3, input_size, forecast_size),
        layers=3,
        layer_size=256
    ))
    # Seasonality block
    blocks.append(NBeatsBlock(
        input_size=input_size,
        theta_size=4*5,  # 5 harmonics
        basis_function=SeasonalityBasis(5, input_size, forecast_size),
        layers=3,
        layer_size=256
    ))
    return NBeats(blocks).to(device)