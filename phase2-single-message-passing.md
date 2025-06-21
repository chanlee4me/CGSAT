# 单轮消息传递实现详解

本文档详细描述了在 CGSAT 项目中实现的单轮消息传递（Single-Round Message Passing）的具体公式和流程。该实现遵循了 `phase2-single-message-passing-design.md` 中定义的设计思路，即结合多头注意力机制和 GRU 进行节点状态的更新。

## 核心组件

- **多头注意力 (Multi-Head Attention)**: 我们采用 `GATv2Conv` 作为注意力的核心实现。相比于标准的 GAT，GATv2 在计算注意力权重时，将查询（Query）、键（Key）和值（Value）都考虑在内，表达能力更强。
- **门控循环单元 (GRU)**: 用于在消息聚合后，结合上一轮的隐藏状态，平滑地更新当前节点的隐藏状态。
- **残差连接 (Residual Connection)**: GRU 的更新机制本身就包含了残差连接的思想，确保了信息的有效流动。

## 消息传递流程

消息传递分为两个阶段：**变量到子句 (V2C)** 和 **子句到变量 (C2V)**。

### 1. 变量到子句 (V2C) 的消息传递

在此阶段，子句节点从其相邻的变量节点收集信息并更新自己的隐藏状态。

#### a. 注意力权重计算与消息聚合

对于一个子句节点 $c$，其邻居变量节点集合为 $\mathcal{N}(c)$。对于每个变量 $v \in \mathcal{N}(c)$，我们首先计算其对 $c$ 的多头注意力消息 $m_{v \to c}$。

令 $h_v, h_c$ 分别为变量 $v$ 和子句 $c$ 的输入特征（或上一轮的隐藏状态）。

$$ a_{vc} = \text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}_q h_c \| \mathbf{W}_k h_v]) $$

其中 $\mathbf{W}_q, \mathbf{W}_k$ 是可学习的线性变换权重矩阵，$\mathbf{a}$ 是单层前馈网络的权重向量。注意力权重 $\alpha_{vc}$ 通过对邻居节点上的 $a$ 值进行 softmax 归一化得到：

$$ \alpha_{vc} = \text{softmax}_v(a_{vc}) = \frac{\exp(a_{vc})}{\sum_{v' \in \mathcal{N}(c)} \exp(a_{v'c})} $$

聚合后的消息 $m_c$ 是所有邻居变量消息的加权和，并通过多头拼接（concat）得到：

$$ m_c = \|_{k=1}^{K} \sum_{v \in \mathcal{N}(c)} \alpha_{vc}^{(k)} \mathbf{W}_v^{(k)} h_v $$

其中 $K$ 是头的数量，$\mathbf{W}_v^{(k)}$ 是第 $k$ 个头的变换矩阵。

#### b. 子句隐藏状态更新

聚合后的消息 $m_c$ 经过层归一化（LayerNorm）和 Dropout 后，送入 GRU 单元，与上一轮的子句隐藏状态 $h_c^{\text{old}}$ 进行融合，得到新的隐藏状态 $h_c^{\text{new}}$：

$$ m'_c = \text{Dropout}(\text{LayerNorm}(m_c)) $$
$$ h_c^{\text{new}} = \text{GRU}(m'_c, h_c^{\text{old}}) $$

### 2. 子句到变量 (C2V) 的消息传递

在此阶段，变量节点从其相邻的子句节点收集信息。特别地，它会使用在 V2C 阶段 **刚刚更新过** 的子句隐藏状态 $h_c^{\text{new}}$ 作为消息来源。

#### a. 注意力权重计算与消息聚合

对于一个变量节点 $v$，其邻居子句节点集合为 $\mathcal{N}(v)$。计算过程与 V2C 类似，但Query和Key的角色互换：

$$ a_{cv} = \text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}_q h_v \| \mathbf{W}_k h_c^{\text{new}}]) $$
$$ \alpha_{cv} = \text{softmax}_c(a_{cv}) $$
$$ m_v = \|_{k=1}^{K} \sum_{c \in \mathcal{N}(v)} \alpha_{cv}^{(k)} \mathbf{W}_c^{(k)} h_c^{\text{new}} $$

#### b. 变量隐藏状态更新

与子句更新类似，聚合后的变量消息 $m_v$ 经过处理后，送入 GRU 更新变量的隐藏状态：

$$ m'_v = \text{Dropout}(\text{LayerNorm}(m_v)) $$
$$ h_v^{\text{new}} = \text{GRU}(m'_v, h_v^{\text{old}}) $$

这样，我们就完成了一轮完整的 V2C 和 C2V 消息传递，得到了所有节点更新后的隐藏状态 $h^{\text{new}}$。
