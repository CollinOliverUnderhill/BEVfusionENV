# 最近改动说明（K-Radar 与雷达并行融合）

## ParallelRadarFuser 模块
- 新增了 `mmdet3d/models/fusers/parallel_radar.py`，提供主模态（相机/激光雷达等）与雷达的**双支路并行融合**。主模态先拼接后经过卷积，雷达走独立卷积分支，最终通道对齐后相加，便于在雷达主导的数据集（如 K-Radar）中注入额外几何线索。
- 通过 `model.fuser.type: ParallelRadarFuser` 即可在配置中启用，并可用 `radar_index` 指定输入列表中雷达特征的位置，`primary_in_channels`/`radar_in_channels`/`out_channels` 控制各支路的通道数。

## K-Radar 数据处理与脚本
- `tools/data_converter/kradar_converter.py` 现在针对「场景编号 → 多模态子目录」的目录结构进行了实现：自动读取 `time_info/time_info.csv`（或直接从 `radar_tesseract`/`os1-128` 文件名推断帧 ID），收集 `cam-front/left/right/rear`、`os1-128`、`os2-64`、`radar_tesseract`、`info_calib`、`info_label_rev2` 等路径，输出 `kradar_infos_<split>.pkl`，在 `tools/create_data.py` 中注册，可通过 `python tools/create_data.py --dataset kradar --root-path <root> --version <split>` 调用。
- 新增 `tools/train_kradar.py` 与 `tools/test_kradar.py` 包装脚本：在重用现有配置时自动重写数据根目录，方便直接在 K-Radar 数据上训练或评测。

## README 补充
- README 新增 K-Radar 支持说明，包含数据准备与调用上述脚本的示例命令，帮助在超算/容器环境下快速启动 K-Radar 的数据预处理与训练/评测流程。
