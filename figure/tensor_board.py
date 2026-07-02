# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
用 TensorBoard 画出 wan/modules/model.py 中 WanModel 的模型架构图。

本脚本把 DiT 主干 (WanModel) 的计算图写入 TensorBoard 的 event 文件，可在
TensorBoard 的 GRAPHS 面板中查看完整的网络架构；同时把各输入/输出张量 shape
记录到 TensorBoard 的 TEXT 面板以及 figure/wan_model_io_shapes.txt。

运行:
    conda activate infworld
    python figure/tensor_board.py                # 默认 2 个 transformer block, 图更清晰
    python figure/tensor_board.py --layers 4     # 自定义 block 数
    python figure/tensor_board.py --full         # 完整 30 层 (需要较大内存)

查看:
    tensorboard --logdir figure/runs
    浏览器打开 http://localhost:6006 的 GRAPHS 面板

------------------------------------------------------------------------------
为什么要做“可追踪 (traceable) 封装”:
  TensorBoard 的 add_graph 底层用 torch.jit.trace 跑一次前向来记录计算图。
  而 WanModel.forward 原生形态无法被直接 trace：
    1) 输入是 List[Tensor]（每个样本 [C, F, H, W]），trace 需要张量/张量元组；
    2) 自/交叉注意力走 flash_attention，要求 CUDA；
    3) rope_apply 里用了复数运算 + grid_sizes.tolist() 的数据依赖循环。
  因此这里：
    * 用 batched 张量输入重写一遍与原版逐层对应的 forward（TraceableWanModel）；
    * 把 flash_attention 换成 scaled_dot_product_attention（CPU/GPU 均可）；
    * 把 rope_apply 换成 identity（旋转位置编码不改变张量 shape / 拓扑结构，
      对“看架构”无影响）。
  除上述两处替换外，patch_embedding / time_embedding / text_embedding /
  WanAttentionBlock / Head / unpatchify 全部复用 wan/modules/model.py 的真实模块，
  因此图中的层次结构、维度与真实模型完全一致。
------------------------------------------------------------------------------
"""

import argparse
import os
import sys
import types

import torch

# ------------------------ 让 `import wan.*` 可用 ------------------------ #
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# 直接加载 wan/modules/model.py，避免触发 wan/__init__.py 与 wan/modules/__init__.py，
# 后者会 import t5.py，而 t5.py 在类定义时调用 torch.cuda.current_device()，
# 在无 GPU 环境下会直接报错。这里为父包注册“桩(stub)包”，仅提供 __path__，
# 使 model.py 内部的相对导入 (from .attention import ...) 仍可正常解析。
for _pkg in ('wan', 'wan.modules'):
    if _pkg not in sys.modules:
        _stub = types.ModuleType(_pkg)
        _stub.__path__ = [os.path.join(_REPO_ROOT, *_pkg.split('.'))]
        sys.modules[_pkg] = _stub

from wan.modules import model as wan_model  # noqa: E402
from wan.modules.model import WanModel, sinusoidal_embedding_1d  # noqa: E402

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
    num_layers=30,    # transformer block 数量 (完整模型)
    window_size=(-1, -1),
    qk_norm=True,
    cross_attn_norm=True,
    eps=1e-6,
)


# ------------------------ trace 友好的替换实现 ------------------------ #
def _sdpa_attention(q, k, v, *args, **kwargs):
    """替换 flash_attention: 用 F.scaled_dot_product_attention。

    输入/输出与 flash_attention 一致: q,k,v 形状均为 [B, L, N, D]，返回 [B, Lq, N, D]。
    忽略变长 mask (q_lens/k_lens) 与 window_size —— 它们只影响数值, 不影响张量
    shape 与网络拓扑, 对“看架构”没有影响。
    """
    qh = q.transpose(1, 2).float()  # [B, N, L, D]
    kh = k.transpose(1, 2).float()
    vh = v.transpose(1, 2).float()
    out = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh)
    return out.transpose(1, 2).to(v.dtype)  # [B, L, N, D]


def _rope_identity(x, grid_sizes, freqs):
    """替换 rope_apply: 恒等映射。

    旋转位置编码只做逐元素相位旋转, 不改变张量 shape, 因此在“结构可视化”里用
    identity 代替, 既能避开复数运算 (torch.jit.trace 不友好) 又不影响图的拓扑。
    """
    return x


class TraceableWanModel(torch.nn.Module):
    """把 WanModel 包装成可被 torch.jit.trace / TensorBoard.add_graph 追踪的形态。

    forward 接收 batched 张量而非 List[Tensor], 其余逐层调用真实 WanModel 的子模块,
    与原始 WanModel.forward 一一对应 (patch -> time/text embedding -> N x block ->
    head -> unpatchify)。
    """

    def __init__(self, model: WanModel):
        super().__init__()
        self.model = model

    def forward(self, x, t, context):
        r"""
        Args:
            x       (Tensor): 视频 latent, 形状 [B, C_in, F, H, W]
            t       (Tensor): 扩散时间步,   形状 [B]
            context (Tensor): 文本特征,     形状 [B, L_text, text_dim]
        Returns:
            Tensor: 去噪后的 latent, 形状 [B, C_out, F, H, W]
        """
        m = self.model

        # 1) patch embedding: Conv3d, 把 latent 切块并投影到 dim 维
        x = m.patch_embedding(x)                       # [B, dim, F, Hp, Wp]
        b, _, f, hp, wp = x.shape
        x = x.flatten(2).transpose(1, 2)               # [B, L, dim], L = F*Hp*Wp

        # 2) time embedding (给每层做 AdaLN 调制)
        e = m.time_embedding(
            sinusoidal_embedding_1d(m.freq_dim, t).float())   # [B, dim]
        e0 = m.time_projection(e).unflatten(1, (6, m.dim))    # [B, 6, dim]

        # 3) text embedding
        context = m.text_embedding(context)            # [B, L_text, dim]

        # 传给各 block 的辅助量 (rope 已被替换为 identity, seq/grid 仅占位)
        seq_lens = torch.full((b,), x.shape[1], dtype=torch.long)
        grid_sizes = torch.tensor([[f, hp, wp]] * b, dtype=torch.long)

        # 4) N x transformer block (self-attn + cross-attn + FFN, 全部 AdaLN 调制)
        for blk in m.blocks:
            x = blk(x, e0, seq_lens, grid_sizes, m.freqs, context, None)

        # 5) 输出头 + 反 patch, 还原成 latent 网格
        x = m.head(x, e)                               # [B, L, C_out*prod(patch)]
        x = self._unpatchify(x, f, hp, wp)             # [B, C_out, F, H, W]
        return x

    def _unpatchify(self, x, f, hp, wp):
        """WanModel.unpatchify 的 batched 版本 (原版逐样本处理 List)。"""
        c = self.model.out_dim
        pf, ph, pw = self.model.patch_size
        b = x.shape[0]
        x = x.view(b, f, hp, wp, pf, ph, pw, c)
        x = torch.einsum('bfhwpqrc->bcfphqwr', x)
        x = x.reshape(b, c, f * pf, hp * ph, wp * pw)
        return x


def build_model(num_layers):
    """构造用于可视化的 WanModel (t2v)。

    维度与真实 1.3B 完全一致, 仅 block 数可调 (默认少几层, 图更清晰、更省内存);
    每个 block 的内部结构与完整模型逐字节相同。
    """
    cfg = dict(WAN_T2V_1_3B_CONFIG)
    cfg['num_layers'] = num_layers
    print(f'正在构造 WanModel (t2v, dim={cfg["dim"]}, num_layers={num_layers}) ...')
    model = WanModel(**cfg)
    model.eval()
    return model


def _example_inputs(model):
    """构造一组小尺寸示例输入 (够展示结构即可, 省内存/加速 trace)。"""
    b = 1
    c_in = model.in_dim
    f, h, w = 4, 16, 16        # latent 时空尺寸 (小尺寸示例)
    text_len = model.text_len  # 512
    text_dim = WAN_T2V_1_3B_CONFIG['text_dim']  # 4096

    x = torch.randn(b, c_in, f, h, w)
    t = torch.tensor([500.0] * b)
    context = torch.randn(b, text_len, text_dim)
    return x, t, context


def _io_shape_report(model, inputs, output):
    x, t, context = inputs
    lines = []
    lines.append('=' * 78)
    lines.append('Wan2.1 1.3B T2V —— DiT 主干 (WanModel) 输入/输出 shape')
    lines.append('=' * 78)
    lines.append('[本次 trace 使用的示例张量]')
    lines.append(f'    输入  x        (video latent) : {tuple(x.shape)}   '
                 f'# [B, C_in={model.in_dim}, F, H, W]')
    lines.append(f'    输入  t         (timestep)    : {tuple(t.shape)}   # [B]')
    lines.append(f'    输入  context   (text feats)  : {tuple(context.shape)}   '
                 f'# [B, text_len={model.text_len}, text_dim=4096]')
    lines.append(f'    输出  x_denoised(latent)      : {tuple(output.shape)}   '
                 f'# [B, C_out={model.out_dim}, F, H, W]')
    lines.append('-' * 78)
    lines.append('[真实推理时的代表性尺寸] 480P(832x480) / 81 帧, 经 Wan-VAE 编码后:')
    lines.append('    latent x      : [1, 16, 21, 60, 104]   '
                 '# F=(81-1)/4+1=21, H=480/8=60, W=832/8=104')
    lines.append(f'    patch_size    : {model.patch_size}   '
                 '# 序列长度 L = F * (H/2) * (W/2) = 21*30*52 = 32760')
    lines.append('    context       : [1, 512, 4096]         # T5(umt5-xxl) 文本特征')
    lines.append('    output latent : [1, 16, 21, 60, 104]   # 与输入 latent 同形状')
    lines.append('-' * 78)
    lines.append('说明: 可视化图中 flash_attention 已替换为 scaled_dot_product_attention,')
    lines.append('      rope_apply 已替换为 identity —— 两者均不改变张量 shape 与网络拓扑。')
    lines.append('=' * 78)
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='用 TensorBoard 画出 WanModel 的模型架构图')
    parser.add_argument('--layers', type=int, default=2,
                        help='可视化用的 transformer block 数 (默认 2, 图更清晰)')
    parser.add_argument('--full', action='store_true',
                        help='使用完整 30 层 (需要较大内存)')
    parser.add_argument('--logdir', type=str,
                        default=os.path.join(_OUT_DIR, 'runs'),
                        help='TensorBoard event 输出目录')
    args = parser.parse_args()

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print('[ERROR] 未安装 tensorboard, 请先: pip install tensorboard')
        sys.exit(1)

    # 关键: 用 trace 友好实现替换注意力与 rope, 使模型可被 torch.jit.trace 追踪
    wan_model.flash_attention = _sdpa_attention
    wan_model.rope_apply = _rope_identity

    torch.manual_seed(0)
    num_layers = 30 if args.full else args.layers
    base_model = build_model(num_layers)
    model = TraceableWanModel(base_model)
    model.eval()

    inputs = _example_inputs(base_model)
    with torch.no_grad():
        output = model(*inputs)

    report = _io_shape_report(base_model, inputs, output)
    print('\n' + report + '\n')

    # 1) 写 TensorBoard 计算图 + shape 文本面板
    writer = SummaryWriter(log_dir=args.logdir)
    with torch.no_grad():
        writer.add_graph(model, inputs)
    writer.add_text('IO_shapes', '```\n' + report + '\n```')
    writer.close()

    # 2) 另存一份 shape 说明到 txt
    txt_path = os.path.join(_OUT_DIR, 'wan_model_io_shapes.txt')
    with open(txt_path, 'w', encoding='utf-8') as fp:
        fp.write(report + '\n')

    print(f'[OK] TensorBoard event 已写入: {args.logdir}')
    print(f'[OK] 输入/输出 shape 已写入: {txt_path}')
    print('\n查看模型架构图:')
    print(f'    tensorboard --logdir {args.logdir}')
    print('    浏览器打开 http://localhost:6006 -> GRAPHS 面板')


if __name__ == '__main__':
    main()
