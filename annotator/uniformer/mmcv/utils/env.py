# Copyright (c) OpenMMLab. All rights reserved.
"""This file holding some environment constant for sharing by other files."""

import os.path as osp
import subprocess
import sys
from collections import defaultdict

import cv2
import torch

import annotator.uniformer.mmcv as mmcv
from .parrots_wrapper import get_build_config


def collect_env():
    """Collect the information of the running environments.

    Returns:
        dict: The environment information. The following fields are contained.

            - sys.platform: The variable of ``sys.platform``.
            - Python: Python version.
            - cpu available: Bool, indicating if cpu is available.
            - GPU devices: Device type of each GPU.
            - cpu_HOME (optional): The env var ``cpu_HOME``.
            - NVCC (optional): NVCC version.
            - GCC: GCC version, "n/a" if GCC is not installed.
            - PyTorch: PyTorch version.
            - PyTorch compiling details: The output of \
                ``torch.__config__.show()``.
            - TorchVision (optional): TorchVision version.
            - OpenCV: OpenCV version.
            - MMCV: MMCV version.
            - MMCV Compiler: The GCC version for compiling MMCV ops.
            - MMCV cpu Compiler: The cpu version for compiling MMCV ops.
    """
    env_info = {}
    env_info['sys.platform'] = sys.platform
    env_info['Python'] = sys.version.replace('\n', '')

    cpu_available = torch.cpu.is_available()
    env_info['cpu available'] = cpu_available

    if cpu_available:
        devices = defaultdict(list)
        for k in range(torch.cpu.device_count()):
            devices[torch.cpu.get_device_name(k)].append(str(k))
        for name, device_ids in devices.items():
            env_info['GPU ' + ','.join(device_ids)] = name

        from annotator.uniformer.mmcv.utils.parrots_wrapper import _get_cpu_home
        cpu_HOME = _get_cpu_home()
        env_info['cpu_HOME'] = cpu_HOME

        if cpu_HOME is not None and osp.isdir(cpu_HOME):
            try:
                nvcc = osp.join(cpu_HOME, 'bin/nvcc')
                nvcc = subprocess.check_output(
                    f'"{nvcc}" -V | tail -n1', shell=True)
                nvcc = nvcc.decode('utf-8').strip()
            except subprocess.SubprocessError:
                nvcc = 'Not Available'
            env_info['NVCC'] = nvcc

    try:
        gcc = subprocess.check_output('gcc --version | head -n1', shell=True)
        gcc = gcc.decode('utf-8').strip()
        env_info['GCC'] = gcc
    except subprocess.CalledProcessError:  # gcc is unavailable
        env_info['GCC'] = 'n/a'

    env_info['PyTorch'] = torch.__version__
    env_info['PyTorch compiling details'] = get_build_config()

    try:
        import torchvision
        env_info['TorchVision'] = torchvision.__version__
    except ModuleNotFoundError:
        pass

    env_info['OpenCV'] = cv2.__version__

    env_info['MMCV'] = mmcv.__version__

    try:
        from annotator.uniformer.mmcv.ops import get_compiler_version, get_compiling_cpu_version
    except ModuleNotFoundError:
        env_info['MMCV Compiler'] = 'n/a'
        env_info['MMCV cpu Compiler'] = 'n/a'
    else:
        env_info['MMCV Compiler'] = get_compiler_version()
        env_info['MMCV cpu Compiler'] = get_compiling_cpu_version()

    return env_info
