import torch
import random
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from vision_transformer import VisionTransformer

def active_head(model: VisionTransformer):
    for name, param in model.named_parameters():
        if "heads" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

def active_full(model: VisionTransformer):
    for param in model.parameters():
        param.requires_grad = True

def activate_random(model: VisionTransformer, activate_ratio: float):
    """randomly select some neurals, and froze other neurals"""
    neuron_groups = {}
    for name, param in model.named_parameters():
        if "encoder.layers" in name:
            # Get the neuron identifier by removing the .weight or .bias suffix
            neuron_key = name.rsplit('.', 1)[0]
            if neuron_key not in neuron_groups:
                neuron_groups[neuron_key] = []
            neuron_groups[neuron_key].append((name, param))
        else:
            param.requires_grad = True

    # Randomly select neuron groups to activate
    num_neurons = len(neuron_groups)
    num_activate = int(num_neurons * activate_ratio)
    activate_neuron_keys = random.sample(list(neuron_groups.keys()), num_activate)

    # 这段代码是有问题的，比如在linear层的param中，一个weight就是一个n维的向量，我们只需要deactivate其中一部分参数
    assert False

    # Set the requires_grad attribute of parameters
    for neuron_key, param_group in neuron_groups.items():
        if neuron_key in activate_neuron_keys:
            for _, param in param_group:
                param.requires_grad = True
        else:
            for _, param in param_group:
                param.requires_grad = False

def activate_based_on_gradient_trace(model: VisionTransformer, activate_ratio: float, val_loader: DataLoader, device):
    """
    .. math::
        I(w_{i}) = |w_{i} \\nabla_{w_{i}} \\mathcal{L}|
    """
    model.to(device)
    model.eval()

    # 存储每个神经元的梯度轨迹
    neuron_gradient_traces = {}

    for data in tqdm(val_loader, ncols=80, desc="calculating gradient traces"):
        images, labels = data[0].to(device), data[1].to(device)

        # 清零梯度
        model.zero_grad()

        # 前向传播
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        # 反向传播计算梯度
        loss.backward()

        # 计算梯度
        for name, param in model.named_parameters():
            if "encoder.layers" in name:
                # 去掉 .weight 或 .bias 后缀得到神经元的标识
                neuron_key = name.rsplit('.', 1)[0]
                if neuron_key not in neuron_gradient_traces:
                    neuron_gradient_traces[neuron_key] = 0

                # 计算梯度轨迹
                if param.grad is not None:
                    # 这里也有问题，见上面random的部分
                    print(name)
                    print(param)
                    print(param.grad)
                    exit(0)
                    gradient_trace = torch.abs(param * param.grad).sum().item()
                    neuron_gradient_traces[neuron_key] += gradient_trace
    exit(0)

    # 按梯度轨迹排序
    sorted_neuron_keys = sorted(neuron_gradient_traces.items(), key=lambda item: item[1], reverse=True)

    # 选择要激活的神经元数量
    num_neurons = len(sorted_neuron_keys)
    num_activate = int(num_neurons * activate_ratio)
    activate_neuron_keys = [key for key, _ in sorted_neuron_keys[:num_activate]]

    # 设置参数的 requires_grad 属性
    for name, param in model.named_parameters():
        if "encoder.layers" in name:
            neuron_key = name.rsplit('.', 1)[0]
            if neuron_key in activate_neuron_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = True