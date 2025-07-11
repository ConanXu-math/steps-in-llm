

# Network and backpropagation





# Bigram

## 一、背景设定

一个**bigram语言模型**，其核心思想是学习从当前字符 $i$ 到下一个字符 $j$ 的条件概率分布 $P(j \mid i)$。训练数据中用 `N[i,j]` 统计了这种 bigram 的出现次数。

然后我们有模型：
$$
\text{logits} = x_{\text{enc}} @ w
$$
其中：

- $x_{\text{enc}} \in \mathbb{R}^{B \times V}$：每一行是 one-hot 编码（当前字符）
- $w \in \mathbb{R}^{V \times V}$：权重矩阵
- 输出 $\text{logits} \in \mathbb{R}^{B \times V}$：预测的每个位置下一个字符的 logits

再经过 softmax 得到概率分布：
$$
P(j \mid i) = \frac{\exp(w_{i,j})}{\sum_{j'} \exp(w_{i,j'})}
$$

------

## 二、训练目标

使用的是交叉熵损失（等价于最大似然）：
$$
\mathcal{L} = -\sum_{i,j} N[i,j] \cdot \log P(j \mid i)
= -\sum_{i,j} N[i,j] \cdot \left( w_{i,j} - \log \sum_{j'} \exp(w_{i,j'}) \right)
$$
这其实是一个 **最大熵建模问题**，最优时满足：
$$
w_{i,j} = \log N[i,j] + \text{常数项（每行）}
$$
### ✅为什么当 $$q(y \mid x)$$ 是 one-hot 时，交叉熵 $$H(q, p) = -\log p(y \mid x) $$？

### 1. 交叉熵的一般定义

交叉熵是两个概率分布之间的距离度量，其定义为：

$$H(q, p) = - \sum_{y} q(y) \log p(y)$$

在监督分类中：

- $$q(y \mid x$$ 表示真实标签的分布（ground truth）
- $$p(y \mid x$$ 表示模型的预测概率分布

### 2. one-hot 分布的情况

假设分类任务中类别为 $$\{0,1,2,3,4\}$$，真实标签为 $$y = 3$$，则：

$$q(y) = \begin{cases} 1 & \text{if } y = 3 \\ 0 & \text{otherwise} \end{cases}$$

这意味着 $$q(y)$$ 是一个 one-hot 向量。

### 3. 套入交叉熵定义

带入公式：

$$H(q, p) = - \sum_y q(y) \log p(y)$$

由于只有 $q(3) = 1$，其他为 0，上式简化为：

$$H(q, p) = - \log p(3)$$

也就是：

$$\boxed{ H(q, p) = -\log p(y_{\text{true}} \mid x) }$$

### 4. 举例说明

假设模型预测为：

- $$p(y=0) = 0.$$
- $$p(y=1) = 0.$$
- $$p(y=2) = 0.$$
- $$p(y=3) = 0.$$
- $$p(y=4) = 0.$$
  -  且真实标签是 $$y = 3$$，则交叉熵为：

$$H(q, p) = -\log 0.5 = 0.6931$$

### 5. 为什么交叉熵这样定义？

- 源于信息论：表示用分布 $$$$ 编码数据实际来自 $$$$ 所需的信息量；
- 当 $$$$ 是 one-hot 时，交叉熵退化为 log loss；
- 最小化交叉熵 ⟺ 最大化对数似然。

### ✅ 总结

在监督学习中，若真实标签是 one-hot 的，则交叉熵损失为：

$$\boxed{ H(q, p) = -\log p(y_{\text{true}} \mid x) }$$

这就是分类问题中常用的交叉熵损失函数形式。



**很多教学或入门代码只先演示 \*bigram\*（二元字符）模型，而不是直接上 \*trigram\*（三元字符）？**

| 角度       | bigram（二元）                      | trigram（三元）                                              | 说明                                                   |
| ---------- | ----------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------ |
| 参数规模   | 仅需 V × V （V≈27 时约 700 个参数） | 朴素实现需 V² × V （≈19 700 个参数）                         | 参数量随 n 指数级增长，内存与计算量成倍上升            |
| 数据稀疏   | 需要观测到所有 (前1, 当前) 组合     | 需要观测到所有 (前2, 当前) 组合，稀疏严重                    | 训练集不够大时，许多三元组合从未出现，概率估计极不稳定 |
| 过拟合风险 | 较低，容易平滑                      | 高，若不做 Kneser–Ney、Laplace 等平滑，模型对罕见组合会过拟合 |                                                        |
| 代码复杂度 | 一个 V × V 权重矩阵即可             | 需把「前两个字符」编码成 V² 维 one-hot 向量或索引，再接 (V²) × V 权重 | 入门教学往往先让读者理解“矩阵乘以 one-hot = 查表”      |
| 收敛速度   | 小数据量也能快速收敛                | 需要更多 epoch 和更大 batch                                  | 同样的迭代次数下，bigram 收敛更快、loss 更稳定         |
| 上下文信息 | 只记住一个历史字符                  | 记住两个历史字符，语境更丰富                                 | 按需权衡：若想捕捉更长依赖，可改用 RNN/Transformer     |

## 1 soft-max 的平移不变性

设
$$
\operatorname{softmax}(x)_i
=\frac{e^{x_i}}{\sum_{j=1}^{C}e^{x_j}}\,,\qquad
x\in\mathbb R^{C}.
$$
取任意常数 $c\in\mathbb R$，记
$$
y = x+c\mathbf 1,\qquad 
\mathbf 1=[1,1,\dots,1]^{\!\top}\in\mathbb R^{C}.
$$

### 证明

$$
\begin{aligned}
\operatorname{softmax}(y)_i
&=\frac{e^{x_i+c}}{\sum_{j}e^{x_j+c}}
= \frac{e^{x_i}\,e^{c}}{e^{c}\sum_{j}e^{x_j}}
= \frac{e^{x_i}}{\sum_{j}e^{x_j}}
=\operatorname{softmax}(x)_i
\quad\forall i.
\end{aligned}
$$

因此
$$
\boxed{\operatorname{softmax}(x)=\operatorname{softmax}(x+c\mathbf 1)
\quad\text{(任意常数 }c).}
$$
取 $c=-m$（即减去最大值）就是常见的数值稳定做法。

------

## 2 log-softmax 也同样不变

$$
\log\operatorname{softmax}(x)_i
= x_i - \log\!\Bigl(\sum_{j}e^{x_j}\Bigr).
$$

对 $y=x+c\mathbf1$：
$$
\begin{aligned}
\log\operatorname{softmax}(y)_i
&= (x_i+c)
     -\log\!\Bigl(\sum_{j}e^{x_j+c}\Bigr) \\[4pt]
&= (x_i+c)
   -\bigl[\log e^{c}+\log\!\bigl(\sum_{j}e^{x_j}\bigr)\bigr] \\[4pt]
&= x_i - \log\!\Bigl(\sum_{j}e^{x_j}\Bigr)
   =\log\operatorname{softmax}(x)_i.
\end{aligned}
$$

------

## 3 交叉熵同样保持不变

对 one-hot 标签 $q$ 与预测 $p=\operatorname{softmax}(x)$ ，交叉熵为
$$
\mathrm{CE}(q,p)
= -\sum_{i} q_i\log p_i.
$$
若把 $x$ 平移为 $y=x+c\mathbf1$，因为 $\log p_i$ 已证明不变，故整个交叉熵同样不变。

------

## 4 为什么选 **减最大值**？

虽然平移可取任意 $c$，但数值上最安全的是
$$
c=-m,\quad m=\max_i x_i,
$$
因为此时所有指数项满足
$$
x_i-m\le 0\;\;(\text{最大为 }0)，
$$
令 $\exp(x_i-m)\in(0,1]$。这样既避免了上溢（overflow），又使得 $\sum_{j}\exp(x_j-m)\le C$ ，从而 $\log$ 也不会得到 $\infty$。

------

### 小结

- soft-max／log-softmax／交叉熵 **对整体平移完全不变**；

- 取 $c=-\max x$ 能把所有指数项压到 $[0,1]$，是最常用的数值稳定技巧；

- 该结论解释了为何在代码实现里常见

  ```
  python
  
  
  复制编辑
  logits = logits - logits.max(dim=1, keepdim=True).values
  ```

  这一行，对数学结果零影响，却大幅降低溢出风险。

# 	Recurrent neural network

## 初始化的重要性

1. **控制激活分布、避免饱和**  
   合理的初始化能让每层线性变换输出落在激活函数的非饱和区间（例如 tanh 的 `[-1,1]` 或 ReLU 的正区间），保证激活函数导数不趋近于零，从而缓解梯度消失或爆炸的问题。

2. **打破对称性**  
   如果所有权重都初始化为相同值（例如零），不同神经元的更新会完全相同，网络无法学习出多样化特征。随机初始化（带适当缩放）可确保每个神经元接收到不同信号，挖掘更丰富的表达能力。

3. **稳定梯度规模**  
   通过 Xavier/Glorot、He 初始化等方法，根据网络深度和激活函数自动缩放初始权重，能让梯度在反向传播时既不过度放大，也不过度衰减，保持稳定的学习步长，提升收敛速度。

4. **加速收敛**  
   “良好”的初始化让网络一开始就处于条件良好的优化区域，用较少的迭代就能显著降低损失；相反，糟糕的初始化会导致训练缓慢、甚至无法收敛。

5. **配合归一化技术**  
   虽然 BatchNorm、LayerNorm 等归一化层能在一定程度上缓解初始化不当的后果，但在这些层之外保持合理的初始权重分布依然至关重要，有助于整个网络更快进入稳定训练状态。

## the buffers of Batch-Norm

1. **评估／推理阶段的稳定性**

   当我们在验证（`val`）或测试（`test`）集上计算损失时，想让结果对每个 batch 的大小或内部样本分布保持一致。若用当前 batch 自己的均值和方差，验证损失就会被 batch 大小和样本差异“噪声”所影响；而使用训练时累积的 running mean/var，则能保证评价结果更稳定、更可复现。

2. **模拟训练时的“期望”归一化效果**
    在训练阶段，BatchNorm 每次都会根据当前 batch 计算一次均值／方差，并用它更新到 running statistics 中：

   ```python
   # training 时大致流程（伪代码）
   batch_mean = x.mean(dim=0);  batch_var = x.var(dim=0)
   running_mean = momentum * running_mean + (1-momentum) * batch_mean
   running_var  = momentum * running_var  + (1-momentum) * batch_var
   ```

   到了评估阶段，我们希望对每个样本都“用同一套”归一化参数（也就是上面累积得到的 running_mean、running_var），这样归一化效果就和训练时的“长期平均”一致。

3. **避免统计偏差（bias）和数值抖动**

   - **小 batch**：当 batch size 很小时，用 batch 自身算出的均值/方差噪声很大，导致每次评估的输出波动。
   - **分布差异**：验证集本身分布可能和训练集略有不同，用训练时的 global statistics 可以减少分布偏差带来的影响。

4. **小结**

- **训练阶段**：BatchNorm 通常用**当前 batch**的均值/方差来做归一化，并同步更新 running statistics。
- **评估／推理阶段**：关闭梯度、切换到 `model.eval()` 后，BatchNorm 会自动改用**running_mean** 和 **running_var**，因为它们更能反映模型在训练中见过的总体数据分布，从而保证评估结果的稳定和一致。

##  learnable 参数：`bngain`和 `bnbias`

在标准化之后，引入可学习的仿射变换：  
$$
y_i = \gamma \,\hat x_i + \beta,
$$
- **\(\gamma\)**（对应代码中的 `bngain`）：对标准化结果进行**尺度缩放**  
- **\(\beta\)**（对应代码中的 `bnbias`）：对标准化结果进行**平移偏置**

这两个参数能够让网络学习到“是否需要恢复原始分布”或“以某种比例/偏移的分布”：

1. 当 $\gamma=1, \beta=0$ 时，输出即为纯标准化结果；  
2. 当$ \gamma $和 $\beta$ 自由学习时，层可以灵活地选择使用标准化、缩放后的或带偏置的激活分布，从而增强模型表达能力。引入 $\gamma,\beta$ 后，BatchNorm 层既能享受归一化带来的训练稳定性，也不丢失表达多样化分布的灵活性。
