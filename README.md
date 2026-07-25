# Fabric Pattern Matcher

面料花型智能匹配工具 — 拍一张照片，在花型库中检索最相似的花型。

## 演示

https://github.com/user-attachments/assets/演示视频.mp4

## 技术架构

```
照片上传 → 裁剪/全图 → Query Expansion (8视图) → CLIP ViT-B/32 → FAISS → 排序输出
```

- **CLIP ViT-B/32** (OpenAI) — 预训练视觉模型，512 维语义嵌入，光照/角度/尺度不变
- **Query Expansion** — 单次搜索自动生成 8 个视图（5 裁剪 + 2 翻转 + 原图），批量 CLIP forward，各视图独立检索后取最高相似度。解决"循环满身拍照 vs 库中单花型"的尺度/偏移不匹配
- **FAISS** (Meta) — IndexFlatIP 内积索引，毫秒级检索，支持 10 万+ 花型库
- **Gradio** — Web UI

每次检索 ~600ms（8 视图批量 CLIP，CPU），不含模型首次加载。

## 快速开始

```bash
# 安装依赖
pip install open-clip-torch faiss-cpu gradio pillow torch

# 启动 GUI
python app.py --fabric-dir dir/fabric

# 打开浏览器 http://127.0.0.1:7860
```

首次运行会下载 CLIP 模型（~300MB），之后使用本地缓存。

## GUI 功能

- 拖拽/点击/粘贴上传照片 → **自动弹出裁剪工具**
- 框选面料区域 → 松手即搜；不框选点 Confirm 用全图搜
- 置信度分级显示（极高/高/中/较低/低）
- 结果数量可选 5~100 条
- **比对灯箱** — 点击结果图弹出两栏对比，滚轮缩放 + 拖拽平移比对细节
- 设置持久化至 `~/.fabric_matcher/settings.json`

## 目录结构

```
fabric/
├── app.py                  # CLIP + FAISS + Gradio GUI + Query Expansion
├── dir/
│   ├── fabric/             # 花型库（PNG/JPG，任意数量）
│   └── photo/              # 测试照片
├── matcher_v3/             # 传统 CV 方案（v1-v5 迭代记录）
├── .fabric_cache/          # CLIP 嵌入缓存（自动生成）
└── 演示视频.mp4
```

## 缓存

- CLIP 嵌入：`.fabric_cache/embeddings.npy`（~11 MB / 5000 张）
- 文件变更（增删花型图）自动失效重建

## 开发历程

1. **v1-v5**: 传统 CV 方案（颜色直方图 + LBP + FFT + Gabor + ORB + POC）
2. **CLIP+FAISS**: 深度学习方案 → 80% Top-100 (490 花型)
3. **CLIP+HSV+POC**: 混合检索实验 → 颜色和 POC 无增益，已移除
4. **CLIP + Query Expansion**: 8 视图批量检索，解决循环满身不匹配问题

## License

MIT
