# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
使用 torchview (from torchview import draw_graph) 画出 wan/modules/model.py 中
WanModel 的模型架构图，输出 SVG。

画两张图:
  1. 单个 transformer block (WanAttentionBlock) -> figure/wan_attention_block.svg
  2. 完整 WanModel (num_layers=2, 避免图片过大)  -> figure/wan_model.svg

运行:
    conda activate infworld
    python figure/use_torchview.py

------------------------------------------------------------------------------
说明:
  torchview.draw_graph 通过前向 hook 记录计算图, 需要跑一次 forward。
  WanModel / WanAttentionBlock.forward 原生依赖:
    1) flash_attention (要求 CUDA) —— 替换为 scaled_dot_product_attention;
    2) rope_apply (复数运算, trace 不友好) —— 替换为 identity。
  两处替换都只改数值, 不改张量 shape 与网络拓扑, 因此画出的结构与真实模型一致。
  其余部分 (patch/time/text embedding / WanLayerNorm / WanSelfAttention /
  WanT2VCrossAttention / FFN / AdaLN modulation / Head) 全部复用真实模块。
  完整模型用 num_layers=2 (而非 30): 每个 block 结构与完整模型逐字节相同,
  只是重复次数不同, 减少层数可避免 SVG 过大、便于查看整体拓扑。
------------------------------------------------------------------------------
"""

import os
import sys
import types

import torch

# ------------------------ 让 `import wan.*` 可用 ------------------------ #
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# 为父包注册“桩(stub)包”, 仅提供 __path__, 避免触发 wan/__init__.py 与
# wan/modules/__init__.py (后者 import t5.py, 在无 GPU 时会报错)。
for _pkg in ('wan', 'wan.modules'):
    if _pkg not in sys.modules:
        _stub = types.ModuleType(_pkg)
        _stub.__path__ = [os.path.join(_REPO_ROOT, *_pkg.split('.'))]
        sys.modules[_pkg] = _stub

from wan.modules import model as wan_model  # noqa: E402
from wan.modules.model import (  # noqa: E402
    WanAttentionBlock, WanModel, sinusoidal_embedding_1d)

_OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# 让 torchview 的图里, 每个 module 方框额外显示"该层的参数(权重)形状"。
#
# torchview 原生只在方框里画激活张量的 input/output shape, 不显示权重 shape。
# 这里用 monkeypatch 猴补 ComputationGraph.get_node_label:
#   - draw_graph 时 ModuleNode 只保存了模块的 id(compute_unit_id);
#   - 我们提前用 model.named_modules() 建立 {id(module): "weight (...), bias (...)"}
#     映射(只取该模块自身直接持有的参数, recurse=False, 避免把子模块参数重复计入);
#   - 在原始 HTML 标签的表格末尾插入一行, 列出这些权重形状。
# 属于改 torchview 内部行为, 若日后升级 torchview 导致标签结构变化需相应调整。
# ============================================================================
from torchview.computation_graph import ComputationGraph  # noqa: E402

_ORIG_GET_NODE_LABEL = ComputationGraph.get_node_label
_PARAM_INFO: dict = {}  # id(module) -> "weight (…), bias (…)"


def _install_param_labels(model: torch.nn.Module) -> None:
    """扫描 model 的所有子模块, 记录每个模块自身直接持有的参数形状。"""
    _PARAM_INFO.clear()
    for module in model.modules():
        parts = [
            f'{pname} {tuple(p.shape)}'
            for pname, p in module.named_parameters(recurse=False)
        ]
        if parts:
            _PARAM_INFO[id(module)] = ', '.join(parts)


def _patched_get_node_label(self, node):
    """在原标签基础上, 给带参数的 module 方框追加一行权重形状。"""
    label = _ORIG_GET_NODE_LABEL(self, node)
    info = _PARAM_INFO.get(getattr(node, 'compute_unit_id', None))
    if info and '</TABLE>' in label:
        # 表格是 5 列(name + input:2列 + shape:2列), 追加一整行跨满 5 列
        extra_row = (
            '<TR><TD COLSPAN="5" ALIGN="LEFT">'
            f'<FONT COLOR="#7a3ea6" POINT-SIZE="10">params: {info}</FONT>'
            '</TD></TR>'
        )
        idx = label.rfind('</TABLE>')
        label = label[:idx] + extra_row + label[idx:]
    return label


ComputationGraph.get_node_label = _patched_get_node_label


# ------------------------ Wan T2V 1.3B block 配置 ------------------------ #
DIM = 1536          # transformer 隐藏维度
FFN_DIM = 8960      # FFN 中间维度
NUM_HEADS = 12      # 注意力头数
EPS = 1e-6

# 完整 WanModel (t2v) 配置, 取自 wan/configs/wan_t2v_1_3B.py
WAN_T2V_1_3B_CONFIG = dict(
    model_type='t2v',
    patch_size=(1, 2, 2),
    text_len=512,
    in_dim=16,
    dim=DIM,
    ffn_dim=FFN_DIM,
    freq_dim=256,
    text_dim=4096,
    out_dim=16,
    num_heads=NUM_HEADS,
    num_layers=30,     # 完整模型层数, 画图时会被覆盖为 2
    window_size=(-1, -1),
    qk_norm=True,
    cross_attn_norm=True,
    eps=EPS,
)


# ------------------------ trace 友好的替换实现 ------------------------ #
def _sdpa_attention(q, k, v, *args, **kwargs):
    """替换 flash_attention: 用 F.scaled_dot_product_attention (CPU 可跑)。

    q,k,v 形状均为 [B, L, N, D], 返回 [B, Lq, N, D]。忽略变长 mask 与 window_size,
    只影响数值, 不影响张量 shape 与网络拓扑。
    """
    qh = q.transpose(1, 2).float()  # [B, N, L, D]
    kh = k.transpose(1, 2).float()
    vh = v.transpose(1, 2).float()
    out = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh)
    return out.transpose(1, 2).to(v.dtype)  # [B, L, N, D]


def _rope_identity(x, grid_sizes, freqs):
    """替换 rope_apply: 恒等映射 (旋转位置编码不改变 shape / 拓扑)。"""
    return x


class TraceableBlock(torch.nn.Module):
    """把单个 WanAttentionBlock 包装成 torchview 可追踪的形态。

    只暴露 (x, e, context) 三个张量输入, 其余 forward 所需的 seq_lens / grid_sizes /
    freqs 在内部按占位量构造 (rope 已替换为 identity, 这些量不参与真实计算)。
    """

    def __init__(self, block: WanAttentionBlock, dim: int):
        super().__init__()
        self.block = block
        self.dim = dim

    def forward(self, x, e, context):
        r"""
        Args:
            x       (Tensor): 输入序列特征, 形状 [B, L, dim]
            e       (Tensor): 时间步调制,   形状 [B, 6, dim]
            context (Tensor): 文本特征,     形状 [B, L_text, dim]
        """
        b, s = x.shape[0], x.shape[1]
        seq_lens = torch.full((b,), s, dtype=torch.long)
        grid_sizes = torch.zeros(b, 3, dtype=torch.long)  # 占位 (rope=identity)
        freqs = torch.zeros(1024, self.dim // NUM_HEADS // 2)  # 占位
        return self.block(x, e, seq_lens, grid_sizes, freqs, context, None)


class TraceableWanModel(torch.nn.Module):
    """把完整 WanModel 包装成 torchview 可追踪的形态。

    forward 接收 batched 张量而非 List[Tensor], 逐层调用真实 WanModel 的子模块,
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
        """
        m = self.model

        # 1) patch embedding: Conv3d, 把 latent 切块并投影到 dim 维
        x = m.patch_embedding(x)                       # [B, dim, F, Hp, Wp]
        b, _, f, hp, wp = x.shape
        x = x.flatten(2).transpose(1, 2)               # [B, L, dim]

        # 2) time embedding (给每层做 AdaLN 调制)
        e = m.time_embedding(
            sinusoidal_embedding_1d(m.freq_dim, t).float())   # [B, dim]
        e0 = m.time_projection(e).unflatten(1, (6, m.dim))    # [B, 6, dim]

        # 3) text embedding
        context = m.text_embedding(context)            # [B, L_text, dim]

        seq_lens = torch.full((b,), x.shape[1], dtype=torch.long)
        grid_sizes = torch.tensor([[f, hp, wp]] * b, dtype=torch.long)

        # 4) N x transformer block
        for blk in m.blocks:
            x = blk(x, e0, seq_lens, grid_sizes, m.freqs, context, None)

        # 5) 输出头 + 反 patch
        x = m.head(x, e)                               # [B, L, C_out*prod(patch)]
        x = self._unpatchify(x, f, hp, wp)             # [B, C_out, F, H, W]
        return x

    def _unpatchify(self, x, f, hp, wp):
        c = self.model.out_dim
        pf, ph, pw = self.model.patch_size
        b = x.shape[0]
        x = x.view(b, f, hp, wp, pf, ph, pw, c)
        x = torch.einsum('bfhwpqrc->bcfphqwr', x)
        x = x.reshape(b, c, f * pf, hp * ph, wp * pw)
        return x


def _render_svg(graph, out_base):
    """把 torchview 的 graph 渲染为 SVG (矢量图, 无损缩放)。"""
    graph.visual_graph.format = 'svg'
    graph.visual_graph.render(filename=out_base, cleanup=True)
    print(f'[OK] 已生成模型架构图 (SVG): {out_base}.svg')


def draw_single_block(draw_graph):
    """画单个 WanAttentionBlock 的架构图。"""
    block = WanAttentionBlock(
        cross_attn_type='t2v_cross_attn',
        dim=DIM,
        ffn_dim=FFN_DIM,
        num_heads=NUM_HEADS,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=EPS,
    )
    block.eval()
    model = TraceableBlock(block, DIM)
    model.eval()

    b, seq_len, text_len = 1, 32, 32
    x = torch.randn(b, seq_len, DIM)
    e = torch.randn(b, 6, DIM)                 # 时间步调制 (float32, block 内有 assert)
    context = torch.randn(b, text_len, DIM)

    print(f'正在用 torchview 追踪单个 WanAttentionBlock (dim={DIM}, '
          f'ffn_dim={FFN_DIM}, num_heads={NUM_HEADS}) ...')
    _install_param_labels(model)   # 让每个 module 方框显示参数形状
    graph = draw_graph(
        model,
        input_data=(x, e, context),
        graph_name='WanAttentionBlock',
        depth=5,
        expand_nested=True,
        save_graph=False,
    )
    _render_svg(graph, os.path.join(_OUT_DIR, 'wan_attention_block'))


def draw_full_model(draw_graph, num_layers=2):
    """画完整 WanModel 的架构图 (默认 num_layers=2, 避免图片过大)。"""
    cfg = dict(WAN_T2V_1_3B_CONFIG)
    cfg['num_layers'] = num_layers
    base_model = WanModel(**cfg)
    base_model.eval()
    model = TraceableWanModel(base_model)
    model.eval()

    # 小尺寸示例输入 (够展示结构即可, 省内存/加速)
    b, c_in = 1, base_model.in_dim
    f, h, w = 4, 16, 16
    text_len = base_model.text_len          # 512
    text_dim = WAN_T2V_1_3B_CONFIG['text_dim']  # 4096
    x = torch.randn(b, c_in, f, h, w)
    t = torch.tensor([500.0] * b)
    context = torch.randn(b, text_len, text_dim)

    print(f'正在用 torchview 追踪完整 WanModel (t2v, dim={DIM}, '
          f'num_layers={num_layers}) ...')
    _install_param_labels(model)   # 让每个 module 方框显示参数形状
    graph = draw_graph(
        model,
        input_data=(x, t, context),
        graph_name='WanModel',
        depth=4,                 # 层级略浅, 避免整图过大
        expand_nested=True,
        save_graph=False,
    )
    _render_svg(graph, os.path.join(_OUT_DIR, 'wan_model'))


def main():
    try:
        from torchview import draw_graph
    except ImportError:
        print('[ERROR] 未安装 torchview, 请先: pip install torchview graphviz')
        sys.exit(1)

    # 关键: 用 trace 友好实现替换注意力与 rope
    wan_model.flash_attention = _sdpa_attention
    wan_model.rope_apply = _rope_identity

    torch.manual_seed(0)
    draw_single_block(draw_graph)
    draw_full_model(draw_graph, num_layers=2)


if __name__ == '__main__':
    main()
