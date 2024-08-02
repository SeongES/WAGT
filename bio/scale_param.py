import torch
import torch.nn as nn

torch.manual_seed(0)
class ScaleParametrization(nn.Module):
    def __init__(self, features_in, features_out):
        super().__init__()

        self.scale_A = nn.Parameter(torch.ones((features_in,1)))
        self.scale_B = nn.Parameter(torch.ones((1, features_out)))

        self.enabled = True

    def forward(self, original_weights):
        if self.enabled:
            # Return W + (B*A)*scale
            return original_weights * torch.matmul(self.scale_A, self.scale_B)
        else:
            return original_weights
        
class ScaleParametrization_bias(nn.Module):
    def __init__(self, features_in):
        super().__init__()
        self.scale_A = nn.Parameter(torch.ones(features_in))

    def forward(self, original_weights):
        return original_weights * self.scale_A
        
        
def layer_parameterization(layer):
    #bias는 건들지 않고 weight, embedding 만 고침
    features_in, features_out = layer.weight.shape
    return ScaleParametrization(
        features_in, features_out
    )

def bias_parameterization(layer):
    #bias는 건들지 않고 weight, embedding 만 고침
    features_in= layer.weight.shape[0]
    return ScaleParametrization_bias(
        features_in
    )


def get_layer_by_name(model, layer_name): #전체 모델 형태에서 하나씩 들어가는 것
    components = layer_name.split('.')
    layer = model
    for component in components[:-1]:  # Exclude the parameter name itself]
        if component.isdigit():
            layer = layer[int(component)]
        else:
            layer = getattr(layer, component)
    return layer

