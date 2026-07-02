# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Wan2.1 1.3B Text-to-Video (T2V) 模型架构图导出脚本。

做了两件事，并把结果分别写入 figure/ 目录下的 txt 文件：
  1. 直接 print(model)            -> wan_t2v_1_3B_print.txt
  2. torchinfo 的 model summary  -> wan_t2v_1_3B_summary.txt

说明:
  - 这里直接用 1.3B 的超参数构造 WanModel(model_type='t2v')，使用随机初始化权重，
    不需要任何 checkpoint，因为我们只关心“网络结构 / 参数量”，而不是具体权重值。
  - 不执行前向推理：WanModel 内部使用 flash_attention（要求 CUDA + list 输入），
    因此 summary 采用“只遍历子模块、统计参数量”的模式（不跑 forward），可在 CPU 上运行。

运行:
    python figure/figure.py
"""

import os
import sys
import types

import torch

# 将仓库根目录加入 import 路径，以便 `import wan.*`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# 直接加载 wan/modules/model.py，避免触发 wan/__init__.py 与 wan/modules/__init__.py：
# 后者会 import t5.py，而 t5.py 在类定义时调用 torch.cuda.current_device()，
# 在无 GPU 环境下会直接报错。这里为父包注册“桩(stub)包”，仅提供 __path__，
# 使 model.py 内部的相对导入 (from .attention import ...) 仍可正常解析。
for _pkg in ('wan', 'wan.modules'):
    if _pkg not in sys.modules:
        _stub = types.ModuleType(_pkg)
        _stub.__path__ = [os.path.join(_REPO_ROOT, *_pkg.split('.'))]
        sys.modules[_pkg] = _stub

from wan.modules.model import WanModel  # noqa: E402

# 输出目录就是本脚本所在的 figure/ 目录
_OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------ Wan T2V 1.3B 配置 ------------------------ #
# 取自 wan/configs/wan_t2v_1_3B.py + wan/configs/shared_config.py
WAN_T2V_1_3B_CONFIG = dict(
    model_type='t2v',
    patch_size=(1, 2, 2),
    text_len=512,
    in_dim=16,        # VAE latent 通道数 (z_dim)
    dim=1536,         # transformer 隐藏维度
    ffn_dim=8960,     # FFN 中间维度
    freq_dim=256,     # 时间步正弦编码维度
    text_dim=4096,    # T5(umt5-xxl) 文本特征维度
    out_dim=16,       # 输出 latent 通道数
    num_heads=12,     # 注意力头数
    num_layers=30,    # transformer block 数量
    window_size=(-1, -1),
    qk_norm=True,
    cross_attn_norm=True,
    eps=1e-6,
)


def build_model():
    """构造 Wan2.1 1.3B T2V 的 DiT 主干（WanModel）。

    使用 `meta` 设备构造：参数只保留 shape，不分配真实存储，因此内存占用极小
    （本环境 cgroup 限制约 2GB，而 1.3B 参数 fp32 实际需 ~5.2GB）。
    我们只关心网络结构与参数量，不做前向/不需要真实权重，meta 设备完全够用。
    """
    print('正在构造 WanModel (t2v, 1.3B, meta device) ... 仅统计结构, 不分配权重')
    with torch.device('meta'):
        model = WanModel(**WAN_T2V_1_3B_CONFIG)
    model.eval()
    return model


def _format_config_header():
    lines = []
    lines.append('=' * 88)
    lines.append('Wan2.1 1.3B Text-to-Video (T2V) — DiT 主干网络 (WanModel)')
    lines.append('=' * 88)
    lines.append('超参数配置:')
    for k, v in WAN_T2V_1_3B_CONFIG.items():
        lines.append(f'    {k:<16}: {v}')
    lines.append('-' * 88)
    return '\n'.join(lines)


def _param_stats(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines = []
    lines.append('参数量统计:')
    lines.append(f'    总参数量    : {total:,} ({total / 1e9:.4f} B / {total / 1e6:.2f} M)')
    lines.append(f'    可训练参数  : {trainable:,} ({trainable / 1e6:.2f} M)')
    lines.append('-' * 88)
    return '\n'.join(lines), total


def dump_print(model):
    """方式一: 直接 print(model) 的模块层级 repr。"""
    out_path = os.path.join(_OUT_DIR, 'wan_t2v_1_3B_print.txt')
    header = _format_config_header()
    stats, _ = _param_stats(model)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(header + '\n')
        f.write(stats + '\n')
        f.write('方式一: print(model)  —— PyTorch 模块层级结构\n')
        f.write('=' * 88 + '\n')
        f.write(str(model) + '\n')
    print(f'[OK] 已写入 (print 结构): {out_path}')
    return out_path


def dump_summary(model):
    """方式二: torchinfo.summary 的逐层参数表（不执行前向）。"""
    try:
        from torchinfo import summary
    except ImportError:
        print('[WARN] 未安装 torchinfo，跳过 summary。可执行: pip install torchinfo')
        return None

    out_path = os.path.join(_OUT_DIR, 'wan_t2v_1_3B_summary.txt')
    # 不传入 input_data/input_size => 不跑 forward，只遍历子模块统计参数量。
    # 这样可在无 CUDA 的环境下工作（flash_attention 需要 CUDA + list 输入）。
    model_stats = summary(
        model,
        depth=6,
        col_names=('num_params', 'params_percent', 'kernel_size', 'trainable'),
        row_settings=('var_names', 'depth'),
        verbose=0,
    )

    header = _format_config_header()
    stats, _ = _param_stats(model)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(header + '\n')
        f.write(stats + '\n')
        f.write('方式二: torchinfo.summary  —— 逐层参数表 (depth=6, 不执行前向)\n')
        f.write('=' * 88 + '\n')
        f.write(str(model_stats) + '\n')
    print(f'[OK] 已写入 (summary 表): {out_path}')
    return out_path


def main():
    torch.manual_seed(0)
    model = build_model()
    dump_print(model)
    dump_summary(model)
    print('完成。输出文件位于:', _OUT_DIR)


if __name__ == '__main__':
    main()
