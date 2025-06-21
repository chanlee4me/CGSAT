# 多轮消息传递实现详解

本文档详细描述了在 CGSAT 项目中实现多轮消息传递（Multi-Round Message Passing）的思路与具体方式。

## 核心思想：迭代式精化

单轮消息传递允许每个节点聚合其直接邻居的信息。然而，对于复杂的图结构和约束（如 SAT 问题），节点需要感知到距离更远的节点状态才能做出准确的推理。多轮消息传递正是为了解决这个问题。

其核心思想是**迭代式地精化（Iterative Refinement）**节点的隐藏状态。通过执行 $T$ 轮消息传递，一个节点可以聚合到其 $T$ 跳邻居内的信息。

## 实现方式

我们的实现主要基于 `EncoderCoreDecoder` 类中已有的 `core_steps` 循环结构。每一轮循环都代表一次完整的消息传递过程。

### 1. 循环结构

在 `EncoderCoreDecoder` 的 `forward` 方法中，我们执行一个循环 `for _ in range(self.steps)`。在循环的每一步，我们都会调用 `self.core` 模块（即 `GraphNet`），它内部封装了我们此前实现的 `SATMessagePassing` 模块。

### 2. 状态传递

- **初始状态**：经过编码器（Encoder）处理后的节点特征 $h^{(0)}$ 作为核心处理模块的初始隐藏状态。
- **迭代更新**：在第 $t$ 轮（$1 \le t \le T$），我们将节点的初始特征 $x$ 与上一轮的隐藏状态 $h^{(t-1)}$ 进行拼接，形成输入：
  $$ x_{\text{in}}^{(t)} = \text{Concat}(x, h^{(t-1)}) $$
  这个输入被送入 `SATMessagePassing` 模块，经过 V2C 和 C2V 两个阶段的计算，产生新的隐藏状态 $h^{(t)}$：
  $$ h^{(t)} = \text{SATMessagePassing}(x_{\text{in}}^{(t)}, \text{edge_index}) $$

### 3. GRU 与残差连接的协同

一个关键的改动在于我们如何处理每一轮的输出。原始的 `EncoderCoreDecoder` 设计可能包含一个加法式的残差连接，形如 $h^{(t)} = h^{(t-1)} + \Delta h$。

然而，我们的 `SATMessagePassing` 模块内部已经使用了 **GRU (Gated Recurrent Unit)** 来更新节点状态：
$$ h^{(t)} = \text{GRU}(\text{message}, h^{(t-1)}) $$
GRU 本身就是一种复杂的门控机制，它能动态地决定上一轮的状态 $h^{(t-1)}$ 有多少信息需要被保留，以及当前收到的消息 `message` 有多少需要被写入。这已经内含了残差连接的思想。

因此，为了避免冗余和不必要的计算，我们**移除了外层的加法残差连接**，直接将 GRU 的输出作为当前轮次的新隐藏状态。即，`EncoderCoreDecoder` 循环中的更新逻辑从 `h_new = h_old + h_delta` 变为了 `h_new = h_delta`（其中 `h_delta` 是 `SATMessagePassing` 的直接输出）。

通过这种方式，我们以一种更优雅、更高效的方式实现了多轮消息传递，充分利用了 GRU 在序列和图数据处理中的强大能力。
