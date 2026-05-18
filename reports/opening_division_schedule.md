# 开题报告：团队分工与任务进度表

## 2.2 团队分工细节

| 成员 |已完成/正在进行的工作 | 
| --- | --- | 
| w | 完成最初项目结构搭建；完成 `nn/` 核心模块开发及相关测试；完成不同隐藏层结构和optimizer的实验对比；完成后续加分项内容开发| 
| L | 负责 MNIST 数据读取、预处理、训练/验证/测试集划分、mini-batch DataLoader；后续承担 Fashion-MNIST 扩展实验的数据接入；配合完成不同 batch size 的实验对比。|
| X| 完成不同learning-rate的实验对比；承担文档整理工作，撰写开题报告，补充公式推导，整理训练曲线、PPT 和演示材料。|

### 协作方式

本项目使用 **Git + GitHub** 进行代码管理和小组协作。

## 2.3 任务进度表

本项目计划按 5 周推进，覆盖“基础 MLP 准确率达标”和两个加分项。

| 周次 | 阶段目标 | 主要任务 | 负责人 | 阶段交付物 |
| --- | --- | --- | --- | --- |
| 第 1 周 | 环境准备与核心原理验证 | 完成项目目录设计；明确 row-major 数据格式约定；实现并测试 `Linear`、`ReLU`、`SoftmaxCrossEntropyLoss`、`SGD`、`Sequential`；使用随机假数据跑通 `forward -> loss -> backward -> update`。 | 同学 A 主责；同学 B/C 学习接口和基础原理 | 项目骨架；`nn/` 核心模块；单元测试；随机数据前后向传播验证结果。 |
| 第 2 周 | 数据模块与基础训练跑通 | 完成 MNIST 数据读取、展平、归一化和 train/validation/test 划分；实现 DataLoader 的 shuffle 和 mini-batch；编写 `train.py` 和 `evaluate.py` 的基础流程；使用小模型 `784 -> 128 -> 10` 验证 loss 能下降、accuracy 能上升。 | 同学 B 主责数据；同学 A 主责训练联调；同学 C 记录过程 | `data/` 模块；基础训练脚本；验证集 accuracy 记录；数据预处理说明。 |
| 第 3 周 | 准确率达标与超参数实验 | 使用主模型 `784 -> 256 -> 128 -> 10` 训练 MNIST；调试学习率、batch size、初始化方法和训练轮数；绘制 Loss 与 Accuracy 曲线；力争测试集 accuracy 达到 95% 以上。 | 同学 A 主责调参；同学 B 负责 batch size 实验；同学 C 整理曲线和表格 | 达标模型；训练曲线；学习率、batch size、网络深度对比实验；阶段性实验分析。 |
| 第 4 周 | 加分项一：BatchNorm | 在 NumPy 框架中实现 BatchNorm 层，完成 forward/backward；加入 `Linear -> BatchNorm -> ReLU` 的网络结构；比较有无 BatchNorm 时的收敛速度、验证集准确率和训练稳定性。 | 同学 A 主责实现；同学 B 协助测试；同学 C 整理对比结果 | `BatchNorm` 层；相关测试；有/无 BatchNorm 的实验曲线和分析。 |
| 第 5 周 | 加分项二与结项交付 | 接入 Fashion-MNIST 数据集，复用自研 MLP 框架完成训练；比较 MNIST 与 Fashion-MNIST 的准确率差异和原因；完善实验报告、演示视频和答辩 PPT；整理最终代码和压缩包。 | 同学 B 主责数据接入；同学 A 主责模型适配与结果检查；同学 C 主责报告和 PPT | Fashion-MNIST 实验结果；最终实验报告；演示视频素材；答辩 PPT；最终提交压缩包。 |
