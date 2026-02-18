# BEVFusion 仓库代码导读（readmechunqiu）

这份文档面向“第一次接触这个仓库”的同学，目标是帮你快速建立**代码地图**：

- 代码分层在哪里；
- 训练/评估从哪里启动；
- 多模态（相机、激光雷达、雷达）在代码里如何汇合；
- 想改模型、改数据集、改配置时应该去哪些目录。

---

## 1. 仓库整体定位

这是 MIT Han Lab 开源的 **BEVFusion** 实现，基于 MMDetection3D 代码风格组织，核心思想是：

1. 把不同传感器特征映射到统一的 BEV（鸟瞰）空间；
2. 在 BEV 空间做融合（而不是点级别早期融合）；
3. 支持多任务（3D 检测 / BEV 语义分割）共享一套范式。

---

## 2. 目录结构速览（你最常改的部分）

```text
.
├── configs/                    # 训练与评估配置（模型结构、数据管线、超参）
├── mmdet3d/
│   ├── apis/                   # train/test 的上层接口
│   ├── datasets/               # 数据集定义、数据增强、pipeline
│   ├── models/                 # 主体模型（编码器、融合器、头部等）
│   ├── ops/                    # 自定义 CUDA / C++ 算子（BEV pooling 等）
│   ├── core/                   # 3D box、points、后处理等基础模块
│   └── runner/                 # 训练 runner 扩展
├── tools/                      # 脚本入口（train.py / test.py / create_data.py 等）
├── docker/                     # Docker 构建文件
├── setup.py                    # 安装与扩展编译入口
└── README.md                   # 官方说明
```

**建议记忆法**：
- “要跑实验”看 `configs/` + `tools/`；
- “要改模型”看 `mmdet3d/models/`；
- “要改数据读取增强”看 `mmdet3d/datasets/`；
- “要查性能瓶颈”看 `mmdet3d/ops/`。

---

## 3. 训练 / 测试主流程（调用链）

### 3.1 你在命令行做的事

通常会执行：

- 训练：`tools/train.py ...`
- 测试：`tools/test.py ...`

以及 README 提到的分布式包装（如 `torchpack dist-run -np 8 ...`）。

### 3.2 代码里的执行路径（高层）

1. `tools/train.py` / `tools/test.py` 读取 yaml 配置；
2. 通过 builder 构建：
   - dataset + dataloader；
   - model（含 encoders / fuser / heads）；
3. runner 循环迭代，调用 model forward；
4. loss 反传（训练）或 decode + eval（测试）。

你可以把它理解为“**配置驱动**”框架：绝大多数结构变化通过 YAML 就能完成。

---

## 4. 模型代码怎么读（mmdet3d/models）

`mmdet3d/models/` 是本仓库最核心的目录，推荐按下面顺序读：

### 4.1 fusion_models/

- `fusion_models/bevfusion.py`：BEVFusion 主干逻辑（多模态特征接入与任务分支组织）。
- `fusion_models/base.py`：融合模型基类。

### 4.2 encoders 对应组件（分散在 backbones / necks / vtransforms）

- `backbones/`：图像、点云、雷达等 backbone。
- `necks/`：FPN/LSS 等 neck，做多尺度特征整合。
- `vtransforms/`：把图像视角特征变到 BEV（如 LSS、Depth-LSS 及变体）。

### 4.3 fusers/

负责把多个模态 BEV 特征融合：

- `add.py`：加和融合；
- `conv.py`：卷积融合；
- `parallel_radar.py`：为雷达单独分支的融合器（README 的 K-Radar 说明里也提到了）。

### 4.4 heads/

任务头：

- `heads/bbox/`：3D 检测头（如 CenterPoint、TransFusion）；
- `heads/segm/`：BEV 分割头。

> 改任务能力（新增检测头/分割头）通常在这里下手。

---

## 5. 数据部分怎么读（mmdet3d/datasets）

重点目录：

- `datasets/nuscenes_dataset.py`：nuScenes 数据集定义；
- `datasets/pipelines/`：数据变换流水线（加载、多帧融合、增强、格式化）；
- `datasets/builder.py`：按配置构建数据集。

理解数据流的关键：

1. 在 `configs/...yaml` 中声明 pipeline；
2. pipeline 的每个步骤在 `datasets/pipelines/*.py` 对应一个算子；
3. 最终打包成模型 forward 需要的字段（图像、点云、标注等）。

---

## 6. 自定义算子（性能关键）

`mmdet3d/ops/` 里包含大量 CUDA/C++ 扩展，尤其关键的是：

- `ops/bev_pool/`：BEV pooling（BEVFusion 论文中的效率关键点）；
- `ops/voxel/`、`ops/iou3d/`、`ops/roiaware_pool3d/` 等：3D 感知常见算子。

这些模块一般由 `setup.py develop` 编译并注册。若你遇到安装报错，优先检查：

- CUDA 与 PyTorch 版本匹配；
- 编译器版本；
- `python setup.py develop` 日志里具体失败的扩展名。

---

## 7. 配置系统（configs）怎么快速定位

`configs/` 按数据集、任务、模态组织，例如：

- `configs/nuscenes/det/...`：检测任务；
- `configs/nuscenes/seg/...`：分割任务；
- 路径里常见 `camera` / `lidar` / `camera+lidar` 标识模态组合。

你可以把配置理解成“实验说明书”：

- 模型结构：`model.*`
- 数据与增强：`data.*`
- 优化器与学习率：`optimizer.*`, `lr_config.*`
- 训练策略：`runner.*`, `checkpoint_config.*`, `evaluation.*`

---

## 8. 给二次开发者的改动建议

### 8.1 想新增一个融合模块

1. 在 `mmdet3d/models/fusers/` 增加新类；
2. 在 `mmdet3d/models/fusers/__init__.py` 注册导出；
3. 在 YAML 把 `model.fuser.type` 改成你的类名；
4. 跑一个最小配置验证张量 shape。

### 8.2 想支持新数据集

1. 参考 `nuscenes_dataset.py` 新建 dataset 类；
2. 实现对应数据转换/预处理脚本（通常在 `tools/`）；
3. 在 pipeline 中补齐该数据集字段；
4. 新增 `configs/<dataset>/...` 验证 train/test。

### 8.3 想加新任务头

1. 在 `mmdet3d/models/heads/` 增加 head；
2. 在 `fusion_models/bevfusion.py` 的输出分支接入；
3. 增加损失与评估接口；
4. 用一个小 batch 做前向 + loss check。

---

## 9. 一条实用阅读路线（1~2 小时）

如果你时间有限，建议按下面顺序：

1. `README.md`：理解支持的任务与训练命令；
2. 一个目标配置（例如 `configs/nuscenes/seg/fusion-bev256d2-lss.yaml`）；
3. `mmdet3d/models/fusion_models/bevfusion.py`：主干 forward；
4. `mmdet3d/models/vtransforms/*.py`：图像到 BEV 的关键；
5. `mmdet3d/models/fusers/*.py`：融合策略；
6. `mmdet3d/models/heads/*`：任务输出；
7. `mmdet3d/datasets/pipelines/*.py`：数据进模型前长什么样。

这样能最快把“配置-数据-模型-输出”闭环串起来。

---

## 10. 常见坑位排查清单

- **安装后找不到扩展算子**：重新执行 `python setup.py develop`，检查编译日志。  
- **训练显存爆炸**：先降 `batch size`、图像分辨率或减少相机数。  
- **配置改了不生效**：确认是否被 base 配置覆盖（若你使用了继承）。  
- **多模态字段缺失报错**：检查 pipeline 的 `Collect3D` / 数据加载步骤字段名是否一致。  
- **指标异常低**：先确认数据预处理版本、标注路径、类别映射、eval 设置一致。

---

## 11. 总结

这个仓库本质上是一个“**以配置驱动的多模态 BEV 感知框架**”：

- `tools/` 提供入口；
- `configs/` 描述实验；
- `mmdet3d/models/` 实现模型；
- `mmdet3d/datasets/` 负责数据；
- `mmdet3d/ops/` 提供高性能算子。

你只要先吃透一条完整链路（例如 nuScenes 的 camera+lidar 检测配置），再横向扩展到其它任务，学习效率会很高。

