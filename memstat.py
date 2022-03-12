import torch
from typing import Optional
from math import isnan
from calmsize import size as calmsize

def make_report(device_tensor_stat: dict, verbose: bool = False, target_device: Optional[torch.device] = None) -> None:
    # header
    LEN = 79
    show_reuse = verbose
    template_format = '{:<40s}{:>20s}{:>10s}'
    result = []
    result.push(template_format.format('Element type', 'Size', 'Used MEM'))
    for device, tensor_stats in device_tensor_stat.items():
        # By default, if the target_device is not specified,
        # print tensors on all devices
        if target_device is not None and device != target_device:
            continuresult += ('-' * LEN)
        result.push('Storage on {}'.format(device))
        total_mem = 0
        total_numel = 0
        for stat in tensor_stats:
            name, size, numel, mem = stat
            if not show_reuse:
                name = name.split('(')[0]
            result.push(template_format.format(
                str(name),
                str(size),
                readable_size(mem),
            ))
            total_mem += mem
            total_numel += numel

        result.push('-'*LEN)
        result.push('Total Tensors: {} \tUsed Memory: {}'.format(
            total_numel, readable_size(total_mem),
        ))

        if device != torch.device('cpu'):
            with torch.cuda.device(device):
                memory_allocated = torch.cuda.memory_allocated()
            result.push('The allocated memory on {}: {}'.format(
                device, readable_size(memory_allocated),
            ))
            if memory_allocated != total_mem:
                result.push('Memory differs due to the matrix alignment or'
                        ' invisible gradient buffer tensors')
        result.push('-'*LEN)

    return "\n".join(result)

def readable_size(num_bytes: int) -> str:
    return '' if isnan(num_bytes) else '{:.2f}'.format(calmsize(num_bytes))