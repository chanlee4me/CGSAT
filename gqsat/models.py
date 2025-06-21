### The code in this file was originally copied from the Pytorch Geometric library and modified later:
### https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/meta.html#MetaLayer
### Pytorch geometric license is below

# Copyright (c) 2019 Matthias Fey <matthias.fey@tu-dortmund.de>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# add by cl 2025-2-16 
#重要改动说明：
# 1. 移除对MetaLayer的依赖，使用MessagePassing基类重构消息传递逻辑
# 2. 显式实现边、节点、全局三个阶段的处理流程
# 3. 调整scatter操作以适配新版PyG
import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, LayerNorm
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn import MessagePassing  # 引入MessagePassing基类
from torch import nn
import inspect
import yaml
import sys

from .message_passing import SATMessagePassing # 导入新的消息传递模块
#added by cl 25-2-16

class SatModel(torch.nn.Module):
    def __init__(self, save_name=None):
        super().__init__()
        if save_name is not None:
            self.save_to_yaml(save_name)

    @classmethod
    def save_to_yaml(cls, model_name):
        # -2 is here because I want to know how many layers below lies the final child and get its init params.
        # I do not need nn.Module and 'object'
        # this WILL NOT work with multiple inheritance of the leaf children
        frame, filename, line_number, function_name, lines, index = inspect.stack()[
            len(cls.mro()) - 2
        ]
        args, _, _, values = inspect.getargvalues(frame)

        save_dict = {
            "class_name": values["self"].__class__.__name__,
            "call_args": {
                k: values[k] for k in args if k != "self" and k != "save_name"
            },
        }
        with open(model_name, "w") as f:
            yaml.dump(save_dict, f, default_flow_style=False)

    @staticmethod
    def load_from_yaml(fname):
        with open(fname, "r") as f:
            res = yaml.load(f, Loader=yaml.Loader)
        return getattr(sys.modules[__name__], res["class_name"])(**res["call_args"])


#创建一个多层感知机（MLP）网络。接受输入维度、输出维度、隐藏层数量、隐藏层大小等参数。
def get_mlp(
    in_size,
    out_size,
    n_hidden,
    hidden_size,
    activation=nn.LeakyReLU,
    activate_last=True,
    layer_norm=True,
):
    arch = []
    l_in = in_size
    for l_idx in range(n_hidden):
        arch.append(Lin(l_in, hidden_size))
        arch.append(activation())
        l_in = hidden_size

    arch.append(Lin(l_in, out_size))

    if activate_last:
        arch.append(activation())

        if layer_norm:
            arch.append(LayerNorm(out_size))

    return Seq(*arch)

# 继承自 SatModel 的类，定义了一个GNN
# GraphNet 可以处理节点、边和全局特征，并通过多层感知机（MLP）进行更新。
# 继承自 SatModel 的类，定义了一个GNN
# GraphNet 可以处理节点、边和全局特征，并通过多层感知机（MLP）进行更新。
class GraphNet(SatModel):
    def __init__(
        self,
        in_dims,  # 输入维度元组(节点,边,全局)
        out_dims,  # 输出维度元组(节点,边,全局)
        independent=False,  # 是否独立处理各组件
        save_name=None,  # 保存文件名
        e2v_agg="sum",  # 边到节点的聚合方式
        n_hidden=1,  # 隐藏层数量
        hidden_size=64,  # 隐藏层大小
        activation=ReLU,  # 激活函数
        layer_norm=True,  # 是否使用层归一化
        use_sat_message_passing=False, # 新增flag以切换模型
        mp_heads=4, # 消息传递模型的参数
        mp_dropout=0.1
    ):
        super().__init__(save_name)
        self.e2v_agg = e2v_agg
        if e2v_agg not in ["sum", "mean"]:
            raise ValueError("Unknown aggregation function.")

        v_in, e_in, u_in = in_dims
        v_out, e_out, u_out = out_dims
        self.independent = independent
        self.use_sat_message_passing = use_sat_message_passing

        # Edge Model
        edge_mlp_in_dim = e_in if independent else e_in + 2 * v_in + u_in
        self.edge_model = get_mlp(edge_mlp_in_dim, e_out, n_hidden, hidden_size, activation=activation, layer_norm=layer_norm)

        # Node Model
        if self.use_sat_message_passing:
            # The new message passing model handles its own logic
            self.node_model = SATMessagePassing(
                in_channels=v_in,
                out_channels=v_out,
                heads=mp_heads,
                dropout=mp_dropout
            )
        else:
            # The old MLP-based model
            node_mlp_in_dim = v_in if independent else v_in + e_out + u_in
            self.node_model = get_mlp(node_mlp_in_dim, v_out, n_hidden, hidden_size, activation=activation, layer_norm=layer_norm)

        # Global Model
        global_mlp_in_dim = u_in if independent else u_in + v_out + e_out
        self.global_model = get_mlp(global_mlp_in_dim, u_out, n_hidden, hidden_size, activation=activation, layer_norm=layer_norm)

    def forward(self, x, edge_index, edge_attr=None, u=None, v_indices=None, e_indices=None):
        row, col = edge_index

        # 1. Edge Update
        if self.edge_model is not None:
            edge_attr_in = edge_attr if self.independent else torch.cat([x[row], x[col], edge_attr, u[e_indices]], 1)
            edge_attr = self.edge_model(edge_attr_in)

        # 2. Node Update
        if self.node_model is not None:
            if self.use_sat_message_passing:
                # The new model expects (x, edge_index) and handles V2C/C2V internally
                x = self.node_model(x, edge_index)
            else:
                # Old model logic
                if self.e2v_agg == "sum":
                    agg_e = scatter_add(edge_attr, col, dim=0, dim_size=x.size(0))
                else: # "mean"
                    agg_e = scatter_mean(edge_attr, col, dim=0, dim_size=x.size(0))

                x_in = x if self.independent else torch.cat([x, agg_e, u[v_indices]], dim=1)
                x = self.node_model(x_in)

        # 3. Global Update
        if self.global_model is not None:
            if self.independent:
                u_in = u
            else:
                # Aggregate node and edge features for the global update
                agg_v = scatter_mean(x, v_indices, dim=0)
                agg_e = scatter_mean(edge_attr, e_indices, dim=0)
                u_in = torch.cat([u, agg_v, agg_e], dim=1)
            u = self.global_model(u_in)

        return x, edge_attr, u

# 图神经网络结构，包括编码器、核心和解码器三个部分。它可以用来处理图数据中的复杂变换。
class EncoderCoreDecoder(SatModel):
    def __init__(
        self,
        in_dims,
        core_out_dims,
        out_dims,
        core_steps=1,
        encoder_out_dims=None,
        dec_out_dims=None,
        save_name=None,
        e2v_agg="sum",
        n_hidden=1,
        hidden_size=64,
        activation=ReLU,
        independent_block_layers=1,
        use_sat_message_passing=False, # 新增flag
        mp_heads=4,
        mp_dropout=0.1
    ):
        super().__init__(save_name)
        # all dims are tuples with (v,e) feature sizes
        self.steps = core_steps
        # if dec_out_dims is None, there will not be a decoder
        self.in_dims = in_dims
        self.core_out_dims = core_out_dims
        self.dec_out_dims = dec_out_dims

        self.layer_norm = True

        self.encoder = None
        if encoder_out_dims is not None:
            self.encoder = GraphNet(
                in_dims,
                encoder_out_dims,
                independent=True,
                n_hidden=independent_block_layers,
                hidden_size=hidden_size,
                activation=activation,
                layer_norm=self.layer_norm,
            )

        core_in_dims = in_dims if self.encoder is None else encoder_out_dims

        self.core = GraphNet(
            (
                core_in_dims[0] + core_out_dims[0],
                core_in_dims[1] + core_out_dims[1],
                core_in_dims[2] + core_out_dims[2],
            ),
            core_out_dims,
            e2v_agg=e2v_agg,
            n_hidden=n_hidden,
            hidden_size=hidden_size,
            activation=activation,
            layer_norm=self.layer_norm,
            use_sat_message_passing=use_sat_message_passing, # 传递flag
            mp_heads=mp_heads,
            mp_dropout=mp_dropout
        )

        if dec_out_dims is not None:
            self.decoder = GraphNet(
                core_out_dims,
                dec_out_dims,
                independent=True,
                n_hidden=independent_block_layers,
                hidden_size=hidden_size,
                activation=activation,
                layer_norm=self.layer_norm,
            )

        pre_out_dims = core_out_dims if self.decoder is None else dec_out_dims

        self.vertex_out_transform = (
            Lin(pre_out_dims[0], out_dims[0]) if out_dims[0] is not None else None
        )
        self.edge_out_transform = (
            Lin(pre_out_dims[1], out_dims[1]) if out_dims[1] is not None else None
        )
        self.global_out_transform = (
            Lin(pre_out_dims[2], out_dims[2]) if out_dims[2] is not None else None
        )

    def get_init_state(self, n_v, n_e, n_u, device):
        return (
            torch.zeros((n_v, self.core_out_dims[0]), device=device),
            torch.zeros((n_e, self.core_out_dims[1]), device=device),
            torch.zeros((n_u, self.core_out_dims[2]), device=device),
        )

    def forward(self, x, edge_index, edge_attr, u, v_indices=None, e_indices=None):
        # if v_indices and e_indices are both None, then we have only one graph without a batch
        if v_indices is None and e_indices is None:
            v_indices = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            e_indices = torch.zeros(
                edge_attr.shape[0], dtype=torch.long, device=edge_attr.device
            )

        if self.encoder is not None:
            x, edge_attr, u = self.encoder(
                x, edge_index, edge_attr, u, v_indices, e_indices
            )

        latent0 = (x, edge_attr, u)
        latent = self.get_init_state(
            x.shape[0], edge_attr.shape[0], u.shape[0], x.device
        )
        for st in range(self.steps):
            latent = self.core(
                torch.cat([latent0[0], latent[0]], dim=1),
                edge_index,
                torch.cat([latent0[1], latent[1]], dim=1),
                torch.cat([latent0[2], latent[2]], dim=1),
                v_indices,
                e_indices,
            )

        if self.decoder is not None:
            latent = self.decoder(
                latent[0], edge_index, latent[1], latent[2], v_indices, e_indices
            )

        v_out = (
            latent[0]
            if self.vertex_out_transform is None
            else self.vertex_out_transform(latent[0])
        )
        e_out = (
            latent[1]
            if self.edge_out_transform is None
            else self.edge_out_transform(latent[1])
        )
        u_out = (
            latent[2]
            if self.global_out_transform is None
            else self.global_out_transform(latent[2])
        )
        return v_out, e_out, u_out
