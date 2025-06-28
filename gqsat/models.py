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
from minisat.minisat.gym.MiniSATEnv import NODE_TYPE_COL, NODE_TYPE_VAR, NODE_TYPE_CLAUSE, HANDCRAFTED_FEATURES_START_COL


class HeteroEncoder(torch.nn.Module):
    """
    An encoder to handle heterogeneous node features (variables and clauses).
    It projects variable and clause features from their original, different-sized
    spaces into a common-sized hidden space.
    """
    def __init__(self, var_feature_dim, clause_feature_dim, hidden_size):
        super().__init__()
        self.var_feature_dim = var_feature_dim
        self.clause_feature_dim = clause_feature_dim
        self.hidden_size = hidden_size

        # Use simple MLPs to encode each node type
        self.var_encoder = get_mlp(var_feature_dim, hidden_size, 1, hidden_size, activate_last=False)
        self.clause_encoder = get_mlp(clause_feature_dim, hidden_size, 1, hidden_size, activate_last=False)

    def forward(self, x):
        # 1. Create masks to identify variable and clause nodes based on the type column
        var_mask = x[:, NODE_TYPE_COL] == NODE_TYPE_VAR
        clause_mask = x[:, NODE_TYPE_COL] == NODE_TYPE_CLAUSE

        # 2. Extract the raw, un-padded features for each node type.
        # The environment pads features to max_dim; we slice them back to their original size here.
        var_features = x[var_mask, HANDCRAFTED_FEATURES_START_COL : HANDCRAFTED_FEATURES_START_COL + self.var_feature_dim]
        clause_features = x[clause_mask, HANDCRAFTED_FEATURES_START_COL : HANDCRAFTED_FEATURES_START_COL + self.clause_feature_dim]

        # 3. Apply the respective encoders. Handle cases where there are no nodes of a certain type.
        encoded_vars = self.var_encoder(var_features) if var_features.shape[0] > 0 else var_features
        encoded_clauses = self.clause_encoder(clause_features) if clause_features.shape[0] > 0 else clause_features

        # 4. Create a new homogeneous tensor and fill it with the encoded features
        x_out = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=torch.float32)
        if encoded_vars.shape[0] > 0:
            x_out[var_mask] = encoded_vars
        if encoded_clauses.shape[0] > 0:
            x_out[clause_mask] = encoded_clauses
        
        return x_out
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
        if self.use_sat_message_passing:
            # The new message passing model only updates node features.
            # It expects (x, edge_index) as input.
            x = self.node_model(x, edge_index)
            # We return edge_attr and u unchanged to maintain the loop signature in the caller.
            return x, edge_attr, u

        # --- Original logic for the old MLP-based model ---
        row, col = edge_index

        # 1. Edge Update
        if self.edge_model is not None:
            edge_attr_in = edge_attr if self.independent else torch.cat([x[row], x[col], edge_attr, u[e_indices]], 1)
            edge_attr = self.edge_model(edge_attr_in)

        # 2. Node Update
        if self.node_model is not None:
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
        var_feature_dim,
        clause_feature_dim,
        edge_in_dim,
        global_in_dim,
        core_out_dims,
        out_dims,
        core_steps=1,
        dec_out_dims=None,
        save_name=None,
        e2v_agg="sum",
        n_hidden=1,
        hidden_size=64,
        activation=ReLU,
        independent_block_layers=1,
        use_sat_message_passing=False,
        mp_heads=4,
        mp_dropout=0.1
    ):
        super().__init__(save_name)
        self.steps = core_steps
        self.core_out_dims = core_out_dims
        self.dec_out_dims = dec_out_dims
        self.layer_norm = True

        # Encoder: Use the new HeteroEncoder for node features.
        # Edge and global features are passed through.
        self.encoder = HeteroEncoder(var_feature_dim, clause_feature_dim, hidden_size)

        # Core: Input dimensions are now based on the encoder's output (hidden_size)
        # and the original edge/global feature dimensions.
        core_in_dims = (hidden_size, edge_in_dim, global_in_dim)
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
            use_sat_message_passing=use_sat_message_passing,
            mp_heads=mp_heads,
            mp_dropout=mp_dropout
        )

        # Decoder: Input is the output of the core, so this remains unchanged.
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

        # 解码器的输入维度变为两倍，因为它拼接了节点自身特征和全局上下文向量
        self.vertex_out_transform = (
            Lin(pre_out_dims[0] * 2, out_dims[0]) if out_dims[0] is not None else None
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
        # 1. 从原始输入中获取变量节点的掩码，以备解码阶段使用
        var_mask = x[:, NODE_TYPE_COL] == NODE_TYPE_VAR

        if v_indices is None and e_indices is None:
            v_indices = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            e_indices = torch.zeros(
                edge_attr.shape[0], dtype=torch.long, device=edge_attr.device
            )

        # 2. 编码节点特征；边和全局特征直接通过
        x = self.encoder(x)

        latent0 = (x, edge_attr, u)
        latent = self.get_init_state(
            x.shape[0], edge_attr.shape[0], u.shape[0], x.device
        )
        for st in range(self.steps):
            # 准备输入：将初始编码特征与上一步的隐藏状态拼接
            x_in = torch.cat([latent0[0], latent[0]], dim=1)

            if self.core.use_sat_message_passing:
                latent = self.core(
                    x_in,
                    edge_index,
                    latent[1],      # 传递当前的边隐藏状态
                    latent[2],      # 传递当前的全局隐藏状态
                    v_indices,
                    e_indices,
                )
            else:
                edge_attr_in = torch.cat([latent0[1], latent[1]], dim=1)
                u_in = torch.cat([latent0[2], latent[2]], dim=1)
                latent = self.core(
                    x_in,
                    edge_index,
                    edge_attr_in,
                    u_in,
                    v_indices,
                    e_indices,
                )

        if self.decoder is not None:
            latent = self.decoder(
                latent[0], edge_index, latent[1], latent[2], v_indices, e_indices
            )

        # --- 上下文感知解码 ---
        final_v_features = latent[0]

        # 3. 从所有变量节点计算全局上下文向量
        var_features = final_v_features[var_mask]

        # 安全检查：如果图中没有变量节点，则使用零向量作为上下文
        if var_features.shape[0] == 0:
            global_context_vector = torch.zeros(u.shape[0], final_v_features.shape[1], device=x.device)
        else:
            var_batch_indices = v_indices[var_mask]
            # 使用 scatter_mean 计算每个图中所有变量特征的均值
            global_context_vector = scatter_mean(var_features, var_batch_indices, dim=0, dim_size=u.shape[0])
        
        # 4. 将计算出的上下文向量扩展，以便与每个节点的自身特征进行拼接
        expanded_context = global_context_vector[v_indices]
        
        # 5. 创建拼接后的增强特征向量
        augmented_v_features = torch.cat([final_v_features, expanded_context], dim=1)
        
        # 6. 将增强后的特征传入最终的变换层以计算输出
        v_out = (
            augmented_v_features
            if self.vertex_out_transform is None
            else self.vertex_out_transform(augmented_v_features)
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
