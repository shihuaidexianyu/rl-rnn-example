# 10.2.4 强化学习：用小型循环神经网络研究学习策略

在第八章中，我们介绍了如何用强化学习模型描述个体根据奖励调整行为的过程。例如，RW 模型根据实际奖励与预期奖励之间的差异更新价值，再将价值转换为选择概率。这类模型的优势在于，它把关于学习机制的假设写成了明确的数学公式。然而，真实个体是否完全按照这些规则学习？如果模型不能充分预测行为，我们又该怎样寻找更合适的更新规则？

上一节利用神经网络研究价值函数与概率权重函数的形状。本节进一步关注一个随时间展开的问题：**经历一次选择及其结果后，个体如何改变内部状态，并据此作出下一次选择？** 我们将以猴子的反转学习任务为例，先比较经典模型与小型循环神经网络的预测，再分析它们如何响应相同的反馈。

这一思路借鉴了 [Ji-An 等（2025）](https://doi.org/10.1038/s41586-025-09142-4)的 tiny RNN 研究。该研究使用只有少量循环单元的网络拟合选择行为，并通过低维状态动力学研究候选学习策略。本节用自行实现的 RW 与两隐藏单元 GRU 展示这一方法：预测比较回答模型是否更好地描述了行为，动力学分析则进一步考察这种描述由怎样的状态更新产生。

## 从反转学习任务开始

设想你经常光顾两家餐厅，起初其中一家更容易提供满意的食物，后来两家的质量发生了变化。你需要根据新的用餐经历调整偏好，但一次不满意的经历未必足以说明这家餐厅已经变差。反转学习任务将类似的问题放进受控实验，用来研究个体如何在保留过去经验和响应新反馈之间取得平衡。

在本案例所用实验中，猴子每次在两个选项之间作出选择，随后获得奖励或没有获得奖励。一个选项的奖励概率为 0.7，另一个为 0.3；每个完整实验块（block）包含 80 次试次，其中发生一次无提示的概率反转，两个选项的奖励概率互换。原实验将反转安排在 block 中部的随机试次，具体位置不固定，也不取决于猴子当时的表现（[Bartolo 等，2020](https://doi.org/10.1371/journal.pcbi.1007514)）。

任务包含两类 block。在 **what block** 中，奖励概率与图像身份有关，图像的左右位置逐试次变化；在 **where block** 中，奖励概率与左右位置有关。整理数据时，前者使用图像选择，后者使用位置选择，将相关的两个选项统一编码为动作 0 和动作 1。奖励编码为 1，无奖励编码为 0。因此，动作 0 并不总是指屏幕左侧。

![反转学习的单试次流程与奖励概率反转](/Users/hongweiqin/Desktop/rl-rnn-example/doc/figures/reversal_task.svg)

**图 10.2.4-1｜反转学习任务。** a，单次试次的呈现、选择与反馈；几何图形代指两幅刺激图像，虚线圈标出所选项。动物只获得所选项的反馈。b，一个完整 block 包含 80 次试次，两个选项的奖励概率在中部无提示地互换一次。图中初始高低概率的归属及反转位置均为示意。what block 的选项按图像编码，where block 按位置编码。

由于奖励具有概率性，选择高奖励概率的选项也可能没有获得奖励。因此，一次无奖励并不能直接说明规则已经反转。模型需要利用此前的选择与反馈预测行为，这为比较不同的历史整合方式提供了研究场景。我们要预测的是猴子实际会选什么，而不是判断哪个选项在实验规则下更优。

本例只使用[公开数据集（Bartolo 与 Averbeck，2022）](https://doi.org/10.17632/p7ft2bvphx.1)中的行为矩阵 `Y`，不使用神经记录。一次 session 对应一个数据文件，一个 session 中有多个 block；`session_name` 与 `block_order` 共同标识一条独立序列。以下代码调用配套项目的读取函数；本节代码均在项目根目录、已安装项目依赖的 Python 环境中运行。

```python
from pathlib import Path
from rl_rnn_example.dataset import load_blocks

# 汇总同一只猴子的多个 session，保留各 block 的边界。
blocks = load_blocks(
    data_dir=Path("data"),
    animal_name="V",
    trial_start=11,
    trial_end=70,
)

print(len(blocks))                    # 96 个完整 block
print(blocks[0]["actions"].shape)      # (60,)
print(blocks[0]["rewards"].shape)      # (60,)
```

猴子 V 的四个 session 共有 96 个完整 block。当前实现从每个 block 中截取原始第 11–70 次，共 60 次；下文以 $t$ 表示这段保留序列中的时刻。两种模型均在保留序列的起点将内部状态设为零，未使用前 10 次试次作预热，在 block 内发生反转时则继续更新状态。这是本例的建模约定，意味着模型不再保留截取起点之前的行为历史。

送入模型的只有动作与奖励。真实任务类型用于预先确定动作编码，反转位置用于标注图片；两者均不作为模型输入。因此，本例研究的是这种编码下的选择更新，并未要求模型同时判断当前属于 what 还是 where 任务。

## 用 RW 模型建立参照

我们先采用具有两个动作价值的 RW 型模型。令 $Q_t(a)$ 表示第 $t$ 次选择之前，模型对动作 $a$ 的价值估计。模型通过 softmax 将价值转换为选择概率：

$$
p_t(a)=\frac{\exp[\beta Q_t(a)]}{\exp[\beta Q_t(0)]+\exp[\beta Q_t(1)]}
$$

其中，逆温度参数 $\beta$ 控制选择对价值差异的敏感程度：$\beta$ 越大，模型越倾向于选择价值较高的动作。猴子随后选择 $a_t$，并获得奖励 $r_t$，被选动作的价值更新为

$$
Q_{t+1}(a_t)=Q_t(a_t)+\alpha[r_t-Q_t(a_t)]
$$

未选动作的价值保持不变。学习率 $\alpha$ 控制新反馈的影响，方括号中的差值就是奖励预测误差。

例如，当前 $Q_t(0)=0.8$、$\alpha=0.2$，如果猴子选择动作 0 后没有获得奖励，则 $Q_{t+1}(0)=0.8+0.2(0-0.8)=0.64$。另一个动作的价值没有变化，下一次选择动作 0 的概率随之降低。这里的两个数值只用于说明更新公式，实际参数需要由行为数据估计。

这个简单例子体现了 RW 的约束：反馈通过预测误差影响价值，学习率固定，只有被选动作的价值改变。拟合 RW，就是在这些假设下寻找最能预测真实选择的 $\alpha$ 和 $\beta$。同一只猴子的多个训练 block 共同确定一套参数，每个 block 内的价值状态则独立演化。

代码中的一个 block 对应一段独立计算。下面将 `rw.py` 中的概率计算与价值更新合并到一个循环，保留核心逻辑。代码中的 `iTemp` 就是公式中的逆温度 $\beta$。

```python
import numpy as np


class RWModel:
    def __init__(self, alpha=0.5, iTemp=5.0):
        self.alpha = alpha
        self.iTemp = iTemp

    def predict_block(self, actions, rewards):
        values = np.zeros(2)  # 每个 block 从零价值开始。
        probabilities = []
        states = []

        for t in range(len(actions)):
            # 先保存当前状态，并预测当前选择。
            states.append(values.copy())
            scores = self.iTemp * (values - values.max())
            weights = np.exp(scores)
            probabilities.append(weights / weights.sum())

            # 再读取真实动作与奖励，只更新被选动作的价值。
            action = actions[t]
            reward = rewards[t]
            prediction_error = reward - values[action]
            values[action] += self.alpha * prediction_error

        return {
            "probabilities": np.array(probabilities),
            "states": np.array(states),
            "final_state": values.copy(),
        }
```

循环中，预测发生在价值更新之前。减去最大价值是 softmax 的数值稳定处理，不改变选择概率。`states` 保存反馈前的价值，`final_state` 另存最后一次反馈后的价值，二者一起构成后续动力学分析所需的完整状态轨迹。

![RW 的价值更新与选择读出](/Users/hongweiqin/Desktop/rl-rnn-example/doc/figures/rw_architecture.svg)

**图 10.2.4-2｜RW 模型。** 当前价值状态 $\mathbf Q_t$ 经 softmax 产生选择概率 $\mathbf p_t$。观察到真实动作 $a_t$ 与奖励 $r_t$ 后，模型按预测误差更新被选动作的价值，未选动作的价值保持不变，得到 $\mathbf Q_{t+1}$。

## 让 GRU 学习状态更新

循环神经网络同样可以用一个随时间变化的内部状态概括行为历史。本例采用门控循环单元（gated recurrent unit，GRU），并将隐藏状态维度设为 2：

$$
\mathbf h_t=(h_{1,t},h_{2,t})^\mathsf T
$$

网络先从当前状态读出两个动作的分数，再转换为选择概率：

$$
\mathbf p_t=\operatorname{softmax}(W_{\mathrm{out}}\mathbf h_t+\mathbf b_{\mathrm{out}})
$$

观察到本次真实动作与奖励后，网络更新状态：

$$
\mathbf h_{t+1}=\operatorname{GRU}_{\theta}([a_t,r_t],\mathbf h_t)
$$

GRU 内部包含两个门。将本次输入记为 $\mathbf x_t=(a_t,r_t)^\mathsf T$，把它与旧状态 $\mathbf h_t$ 上下拼接，得到一个四维列向量。两个门分别计算为

$$
\boldsymbol{\rho}_t=\sigma\!\left[W_\rho
\begin{pmatrix}\mathbf x_t\\\mathbf h_t\end{pmatrix}
+\mathbf b_\rho\right],\qquad
\mathbf z_t=\sigma\!\left[W_z
\begin{pmatrix}\mathbf x_t\\\mathbf h_t\end{pmatrix}
+\mathbf b_z\right]
$$

$W_\rho$ 与 $W_z$ 是各自的 $2\times4$ 权重矩阵，$\mathbf b_\rho$ 与 $\mathbf b_z$ 是二维偏置。每个门先做矩阵乘法、加上偏置，再通过 sigmoid 函数 $\sigma$，得到两个介于 0 与 1 之间的数。

重置门调节旧状态对候选状态的贡献。按照本项目所用 GRU 的计算方式，候选状态为

$$
\widetilde{\mathbf h}_{t+1}=\tanh\!\left[
W_x\mathbf x_t+\mathbf b_x
+\boldsymbol{\rho}_t\odot(W_h\mathbf h_t+\mathbf b_h)
\right]
$$

其中，$\odot$ 表示逐元素相乘。重置门作用在旧状态经过线性变换后的分支上：某个分量接近 0 时，这部分历史信息对候选状态的贡献较小。当前输入则通过另一条分支参与候选状态的计算。

更新门再将旧状态与候选状态组合起来：

$$
\mathbf h_{t+1}=\mathbf z_t\odot\mathbf h_t
 +(1-\mathbf z_t)\odot\widetilde{\mathbf h}_{t+1}
$$

更新门的某个分量接近 1 时，对应维度更多保留旧状态；接近 0 时，更多采用候选状态。两个门都会随当前输入和历史状态改变，因此同一种无奖励反馈可以产生不同的状态更新。

![GRU 的时间结构、重置门、更新门与候选状态计算](/Users/hongweiqin/Desktop/rl-rnn-example/doc/figures/gru_architecture.svg)

**图 10.2.4-3｜GRU 的时间结构与内部计算。** a，同一组参数 $\theta$ 在各试次重复使用，当前预测 $\mathbf p_t$ 从 $\mathbf h_t$ 读出，本次输入 $\mathbf x_t=(a_t,r_t)^\mathsf T$ 用于更新下一状态。b，两个门的方框直接列出矩阵运算，括号内上下排列的 $\mathbf x_t$ 与 $\mathbf h_t$ 表示向量拼接。门的两部分权重并列、两项偏置相加，仅为书写简便。重置门调节旧状态变换后的贡献，更新门控制旧状态与候选状态的混合。$\odot$ 表示逐元素乘，圆圈中的加号表示逐元素加；隐藏状态及两个门均为二维向量。

RW 和 GRU 的预测都遵守同样的时间顺序。虽然代码把动作与奖励放在同一个输入数组中，预测 $a_t$ 时仍然只能使用此前的历史。配套实现将初始零状态放在状态序列开头，使第一个保留试次由初始状态预测，第二个试次由读完第一条反馈后的状态预测，以此类推。它不会先读取当前动作，再“预测”这个动作。

`gru.py` 中负责模型建立与前向计算的核心部分如下。输入 `inputs` 的形状为 `(T, B, 2)`：`T` 是试次数，`B` 是同时处理的 block 数，最后两项为动作与奖励。每个 block 都有独立的初始状态。

```python
import torch
from torch import nn


class GRUModel(nn.Module):
    def __init__(self, hidden_dim=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_size=2, hidden_size=hidden_dim)
        self.linear = nn.Linear(hidden_dim, 2)

    def forward(self, inputs):
        # 第一维 1 表示一层 GRU，第二维对应不同 block。
        h0 = torch.zeros(
            1, inputs.size(1), self.hidden_dim,
            dtype=inputs.dtype, device=inputs.device,
        )
        updated_states, final_state = self.gru(inputs, h0)

        # GRU 输出的是读完本次输入后的状态，需要向后错开一位。
        # 第一条预测使用 h0，此后依次使用读完上一条输入后的状态。
        states = torch.cat([h0, updated_states[:-1]], dim=0)
        logits = self.linear(states)

        return {
            "logits": logits,
            "states": states,
            "final_state": final_state.squeeze(0),
        }
```

`nn.GRU` 内部完成图中两个门与候选状态的计算，`nn.Linear` 将隐藏状态转换成两个动作分数。这里最关键的一行是 `torch.cat([h0, updated_states[:-1]], dim=0)`：它将状态与预测目标对齐，保证模型预测当前动作时还没有读到该动作。评估时对 `logits` 使用 softmax 得到概率；训练时直接将 `logits` 交给交叉熵损失，保留梯度以更新权重。

RW 与 GRU 都是二维状态模型，但这不意味着它们同样简单。RW 只拟合两个参数，本例 GRU 连同线性读出层共有 42 个可训练参数。较小的状态维度便于画出状态空间；较多的参数则使 GRU 能表示更灵活的更新规则。隐藏维度、参数数量和训练试次数描述的是不同的事情。

这里还需区分两种“学习”。**拟合时，网络通过监督学习调整权重，使预测接近猴子的真实选择；拟合完成后，权重保持固定，隐藏状态随每次动作与奖励改变。** 前者是模型训练，后者用来描述猴子在任务中的学习过程。与训练网络完成工作记忆任务的示例相比，本例的目标是拟合个体行为；奖励是输入信息，优化目标是选择预测，而不是网络自己获得的奖励总量。

## 在相同测试数据上比较预测

为了比较两种模型，我们按完整 block 划分数据，保留各 block 内的试次顺序。同一个 block 只能属于训练、验证、测试中的一个集合。训练集用于拟合参数，验证集用于选择 GRU 的训练停止时机，测试集用于最后的预测比较。

```python
from rl_rnn_example.training import split_blocks

train_blocks, validation_blocks, test_blocks = split_blocks(
    blocks,
    validation_fraction=0.1,
    test_fraction=0.1,
    seed=0,
)

print(len(train_blocks), len(validation_blocks), len(test_blocks))
# 78 9 9
```

这次划分包含 78 个训练 block、9 个验证 block 和 9 个测试 block，分别对应 4680、540 和 540 次试次。模型按猴子分别拟合，但同一只猴子的多个 session 可以共同用于训练；并非一个 session 或一个 block 各拟合一套权重。

两种模型均以提高真实选择的预测概率为目标。评价指标采用每试次平均负对数似然（negative log-likelihood，NLL）：

$$
\operatorname{NLL}=-\frac{1}{N}\sum_{t=1}^{N}\ln p_t(a_t)
$$

其中，$N$ 为所评估试次的总数。NLL 越小，表示模型整体上赋予真实选择的概率越高。选择准确率则检查概率最大的动作是否与真实动作相同。NLL 还区分预测的确信程度：给真实动作分配 0.9 和 0.6 的概率，虽然都可能预测正确，损失却不同。

配套代码把拟合步骤封装为两个函数，便于在相同数据上调用。下文导入项目中的完整模型类；前面的代码仅展示核心计算，省略了部分评估与动力学接口。

```python
import torch
from rl_rnn_example.rw import RWModel
from rl_rnn_example.gru import GRUModel
from rl_rnn_example.training import fit_rw, fit_gru

rw_model = RWModel()
fit_rw(rw_model, train_blocks)

torch.manual_seed(0)
gru_model = GRUModel(hidden_dim=2)
history = fit_gru(
    gru_model,
    train_blocks,
    validation_blocks,
    learning_rate=0.005,
    max_epochs=1000,
    patience=100,
)
```

`fit_rw` 优化 $\alpha$ 与 $\beta$；`fit_gru` 通过反向传播更新网络参数，并恢复验证 NLL 最低时的权重。这里的 `learning_rate` 控制优化器调整权重的步长，与 RW 中描述反馈影响的 $\alpha$ 含义不同。测试数据不参与这两个拟合过程。

下面直接读取项目已保存的拟合结果并重新评价，因此后续图像始终对应同一套参数。`records` 保存各测试 block 的行为和模型状态，可作为 `evaluate` 所需的 block 列表使用。

```python
from rl_rnn_example.analysis import load_results
from rl_rnn_example.metrics import evaluate

models, records, experiment = load_results(
    Path("results/monkey_V_seed0")
)

for name, model in models.items():
    scores = evaluate(model, records)
    print(name, scores["nll"], scores["accuracy"])
```

![RW 与 GRU 在相同测试集上的预测表现](/Users/hongweiqin/Desktop/rl-rnn-example/doc/figures/model_comparison.svg)

**图 10.2.4-4｜测试集上的选择预测。** a，每试次平均 NLL，越低越好。b，选择预测准确率，越高越好。两个模型均评价猴子 V 的同一组 9 个测试 block，共 540 次试次。图中为一次固定数据划分的描述性结果，未作统计推断。

本次拟合得到 RW 的 $\alpha=0.445$、$\beta=3.387$。在测试集上，RW 的 NLL 为 0.401，GRU 为 0.367；准确率分别为 82.2% 和 84.1%。因此，在这次比较中，两隐藏单元 GRU 比当前 RW 基线更好地预测了猴子 V 的选择。

这个结果的范围也需要明确。测试 block 未用于拟合，但它们所在的 session 也有其他 block 出现在训练集中，因此这里评价的是同一只猴子、已有 session 中留出 block 的预测，不能据此声称能泛化到新的 session 或新的个体。本例采用一次固定划分，也只比较了一个经典基线；如果要得出更广泛的模型优劣结论，还需比较其他经典模型，并考察不同划分、初始化和个体下的稳定性。

## 用整个测试集观察状态动力学

更好的预测并没有自动告诉我们模型采用了什么策略。接下来固定已经拟合的参数，让两个模型分别读取全部测试 block 中的真实行为，记录每次反馈前后的状态。这里分析的是**固定模型的状态如何变化**，而不是训练过程中权重怎样变化。

对于 RW，状态是 $\mathbf s_t=(Q_t(0),Q_t(1))^\mathsf T$；对于 GRU，状态是 $\mathbf s_t=(h_{1,t},h_{2,t})^\mathsf T$。二维状态可以表示为平面上的一个点，每次更新就是从 $\mathbf s_t$ 移动到 $\mathbf s_{t+1}$。如果把每个 block 的点按时间连接起来，得到的是各自的状态轨迹；但要比较更新规律，还需要区分是什么反馈引发了这些移动。

动作和奖励均为二值，因此共有四种输入条件：$(0,0)$、$(0,1)$、$(1,0)$、$(1,1)$。固定一种条件后，模型的更新可写为

$$
\mathbf s_{t+1}=F_{a,r}(\mathbf s_t),\qquad
\Delta\mathbf s=F_{a,r}(\mathbf s)-\mathbf s
$$

这样，每个模型都对应四个条件更新函数。实际执行任务时，模型随着输入改变而在这四种更新之间切换。我们将所有测试 block 中属于同一种输入的更新放入同一个面板，从而观察它在不同历史状态下产生的影响。

```python
from rl_rnn_example.analysis import compute_model_dynamics
from rl_rnn_example.plotting import plot_dynamics_comparison

fields = {}
for name, model in models.items():
    limits = (0.0, 1.0) if name == "RW" else (-1.0, 1.0)
    fields[name] = compute_model_dynamics(
        model=model,
        records=records,
        model_name=name,
        grid_size=41,
        state_limits=limits,
        max_arrows_per_condition=None,  # 保留全部测试事件。
    )

plot_dynamics_comparison(fields, Path("results/chapter_figures"))
```

函数先在每个 block 内计算状态更新，再按输入条件汇总。它不会把前一个 block 的末状态与下一个 block 的初状态连接起来。每个 block 的最后一次反馈也有更新后的末状态，因此 9 个 block 的 540 次反馈都能进入分析。

![RW 与 GRU 在整个测试集上的条件状态动力学](/Users/hongweiqin/Desktop/rl-rnn-example/doc/figures/dynamics_comparison.svg)

**图 10.2.4-5｜整个测试集的条件动力学。** 上排 a–d 为 RW，下排 e–h 为 GRU；四列依次为选择 0 无奖励、选择 0 有奖励、选择 1 无奖励、选择 1 有奖励。黑箭头表示真实行为驱动的模型状态更新，四列分别包含 101、188、116、135 次事件；每个模型共显示 540 次更新。背景色及细等值线表示在网格状态上施加该列输入时的更新幅度 $\|\Delta\mathbf s\|_2$。同一模型四个面板共用色标，两种模型的色标独立。橙色虚线为 $P(a=0)=0.5$ 的边界。图中包含全部 9 个测试 block，而非单个 block 的轨迹。

阅读这张图时，可以依次区分三类信息。

首先看**黑箭头**。箭尾是反馈前的模型状态，箭头指向反馈后的状态。同一面板中的箭头可能来自不同 block 或相隔很远的试次，它们共有的是输入条件，并不构成一条按时间排列的路径。箭头来自真实行为驱动的模型计算，坐标本身并不是实验测得的神经活动。

其次看**背景色与细等值线**。为了查看真实状态附近的更新规律，代码在覆盖测试状态范围的矩形区域内建立网格，对每个网格点施加同一种输入，计算 $\|F_{a,r}(\mathbf s)-\mathbf s\|_2$。颜色偏黄表示更新幅度较大，偏紫表示较小；同一条细线上的更新幅度相同。它们类似地形图中的等高线，但这里的“高度”是更新幅度。细线不表示状态会沿线运动，颜色也不表示选择概率；网格中的部分状态可能从未在测试行为中出现。

最后看**橙色虚线**。它表示两个动作的预测概率相等，连接状态更新与行为偏好。对 RW 而言，$Q_t(0)=Q_t(1)$ 时两动作等概率，因此边界正好是对角线；线下方 $Q_t(0)>Q_t(1)$，模型更偏向动作 0。若状态更新跨过边界，概率最大的动作就改变了；即使没有跨线，选择概率也可能发生变化。

GRU 的边界由读出层决定。令两个动作的读出分数之差为

$$
d(\mathbf h)=(\mathbf w_0-\mathbf w_1)^\mathsf T\mathbf h+(b_0-b_1),
\qquad P(a=0\mid\mathbf h)=\frac{1}{1+e^{-d(\mathbf h)}}
$$

边界就是 $d(\mathbf h)=0$。当前权重给出的分数差约为 $-2.060h_1+2.073h_2+0.132$，所以图中的虚线虽然接近对角线，却不是严格的 $h_1=h_2$；在线上方，模型更偏向动作 0。所谓“读出方向”，就是使 $d(\mathbf h)$ 增大的方向，在这里由向量 $(\mathbf w_0-\mathbf w_1)$ 决定。沿与边界平行的方向移动可以改变隐藏状态，却不改变当前读出概率。因此，状态移动的距离本身不足以说明选择偏好改变了多少。

### 两种模型的更新有何不同？

先看 RW 的四个面板。选择动作 0 时，只有横坐标 $Q_0$ 更新：无奖励时向左移动，有奖励时向右移动。选择动作 1 时，只有纵坐标 $Q_1$ 更新：无奖励时向下，有奖励时向上。更新幅度为 $\alpha|r-Q_t(a)|$，完全由被选动作的当前价值及奖励决定。因此，相同 $Q_0$ 下的动作 0 更新幅度相同，形成竖直等值线；动作 1 的等值线则是水平的。图中的几何结构可以直接从 RW 公式推出。

再看 GRU。箭头可以同时改变两个隐藏坐标，方向与幅度也随所在位置改变；等值线呈现弯曲形状。以“选择 0 并获得奖励”的 f 面板为例，处于右下区域的状态会向左上方移动，而已经处于左上区域的部分状态变化较小。结合当前读出层可知，向左上方移动通常会增加选择动作 0 的概率。这展示了网络如何把同一种反馈转换为依赖当前状态的更新。

这样的分析使我们能够在真实行为访问过的区域，检查相同反馈在不同历史状态下如何改变下一次的选择概率。RW 的更新同样依赖当前价值；这里要比较的是这种依赖能否由固定学习率、仅更新被选动作的规则概括。不过，两个隐藏坐标是共同参与读出的，不能直接把 $h_1$ 和 $h_2$ 命名为两个动作的 Q 值，也不能仅因两个坐标都发生变化，就断言猴子更新了未选动作的价值。

跨模型比较时，更合适的共同尺度是概率变化

$$
\Delta p_0=p_0(\mathbf s_{t+1})-p_0(\mathbf s_t)
$$

它描述同一次反馈使模型对动作 0 的偏好增强还是减弱。相反，Q 值与隐藏状态的坐标单位不同，不能通过两行色彩的深浅或箭头长度判断哪一个模型“学习更快”。等值线弯曲或箭头局部汇聚也不能单独证明存在吸引子；如果要提出这一解释，还需检验固定点及其稳定性。

## 从网络更新提出可检验的认知假设

在第八章中，我们先规定价值更新公式，再用行为估计参数。本节增加了另一条研究路径：先用一个能够预测行为的小型网络学习状态更新，再把其中的规律转化为可以检验的认知假设。例如，相同的无奖励反馈是否会因先前积累的证据不同而产生不同影响？一个固定学习率能否概括这些变化？这些问题可以进一步落实为带有明确机制约束的模型，并在独立数据上与原有模型比较。

本例完成了这条路径的前两步：GRU 在当前测试集上的预测优于 RW；条件动力学图展示了两个模型如何处理同一组反馈。预测差异并不自动确定其原因，图中的状态依赖也还不是对动物真实机制的证明。要把观察到的更新规律解释为认知策略，需要验证它是否稳定，并检验由此建立的机制模型是否改善预测。

本节的实现也保留了教学上的简化：采用动作与奖励两个输入、普通线性读出层，以及一次固定的数据划分。它借鉴了 tiny RNN 的研究方法，并非原论文模型与图 4 的逐项复现。小型网络在这里的作用，是提供一个既能拟合行为、又能在低维空间中检查的计算系统，帮助我们把“模型预测得更好”推进为“哪些更新假设值得进一步检验”。

## 参考文献

Ji-An, L., Benna, M. K., & Mattar, M. G. (2025). Discovering cognitive strategies with tiny recurrent neural networks. *Nature, 644*, 993–1001. [doi:10.1038/s41586-025-09142-4](https://doi.org/10.1038/s41586-025-09142-4)

Bartolo, R., Saunders, R. C., Mitz, A. R., & Averbeck, B. B. (2020). Dimensionality, information and learning in prefrontal cortex. *PLOS Computational Biology, 16*(4), e1007514. [doi:10.1371/journal.pcbi.1007514](https://doi.org/10.1371/journal.pcbi.1007514)

Bartolo, R., & Averbeck, B. (2022). *Spike count data for studying dimensionality, information and learning in prefrontal cortex* [Data set, Version 1]. Mendeley Data. [doi:10.17632/p7ft2bvphx.1](https://doi.org/10.17632/p7ft2bvphx.1)
