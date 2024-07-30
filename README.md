# curling-game
冰壶游戏
本实验基于gymnasium构建冰壶场地环境，定义包含冰壶与目标点，其中冰壶在每个决策间隔后选择一个方向施加5单位力，冰壶在运行过程中收到空气阻力（其中注意空气阻力计算时间间隔为0.01s），如果碰到边界会以0.9倍系数镜像反弹，动作空间为上下左右4个，每步通过step()计算reward(-欧式距离)、位置、本轮游戏是否结束。在每轮游戏结束后冰壶与目标重置。
实验中，使用的核心方法为深度Q网络，使用三层全连接神经网络，每层后接relu近似Q函数，输入为游戏内的特征向量，包括冰壶、目标坐标与速度。输出为四个动作的Q值。
DQN的原理为因为最终策略的Q函数满足贝尔曼方程：
<br>
<p align='center'>
$$Q^\pi\left(s,a\right)=r+\gammaQ^\pi\left(s^\prime,\pi\left(s^\prime\right)\right)$$
</p>
则使用迭代的方法通过减小误差δ逼近真正的Q函数：
<br>
<p align='center'>
$$\delta=Q\left(s,a\right)-\left(r+\gamma\max_a^\prime{Q}\left(s^\prime,a\right)\right)$$
</p>
而损失函数使用huber loss，其特点为当误差较小时为二次方，误差较大时为线性，比较平缓：
<br>
<p align='center'>
$$\mathcal{L}=\frac{1}{\left|B\right|}\sum_{\left(s,a,s^\prime,r\right)\inB}\mathcal{L}\left(\delta\right)$$
</p>
为了平衡样本的相关性（可能存在连续样本之间存在相关导致学习了错误的经验）采用了经验回放机制来存储并随机抽样过去的经验。用一个大小为10000的经验回放缓冲区来存储游戏的状态转移信息（2000）。
```python
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
```
在训练过程中，随机抽样一批数据用于更新神经网络模型。

![冰壶](https://github.com/ddsk1/curling-game/blob/main/%E5%86%B0%E5%A3%B6.gif)
