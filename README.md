# Fabric Pattern Matcher

面料花型智能匹配工具 — 拍一张照片，在花型库中检索最相似的花型。

## 技术架构

```
照片上传 → CLIP ViT-B/32 提取嵌入 → HSV颜色直方图融合 → FAISS 向量检索 → POC 结构验证 → 排序输出
```

- **CLIP** (OpenAI) — 预训练视觉模型，提取 512 维语义嵌入，光照/角度/尺度不变
- **HSV 颜色直方图** — 8×4×2=64 维，H 分量光照不变，与 CLIP 加权融合解决色差不匹配
- **POC 相位相关** (Phase-Only Correlation) — 纯 numpy FFT，对最终结果做结构/纹理重合度验证，花型匹配的关键信号
- **FAISS** (Meta) — 向量索引，毫秒级检索，支持 10 万+ 花型库
- **Gradio** — Web UI，拖拽上传即用

每次检索耗时 ~50ms（不含模型首次加载），POC 结构验证 +~30ms。

## 快速开始

```bash
# 安装依赖
pip install open-clip-torch faiss-cpu gradio pillow

# 启动 GUI
python app.py --fabric-dir dir/fabric

# 打开浏览器 http://127.0.0.1:7860
```

首次运行会下载 CLIP 模型（~300MB），之后使用本地缓存。

## GUI 功能

- 拖拽/点击/粘贴上传照片
- **裁剪工具** — 框选面料区域，去除背景干扰
- **Color 滑块** — 调节颜色匹配权重（0=纯CLIP，1=纯颜色），拖拽即重新搜索
- **结果数量可选** 5~100 条
- 置信度分级显示（极高/高/中/较低/低）
- **比对灯箱** — 点击结果图弹出两栏对比，左侧支持滚轮缩放 + 拖拽平移比对细节
- 设置持久化至 `~/.fabric_matcher/settings.json`

## 缓存

- CLIP 嵌入：`.fabric_cache/embeddings.npy`（~11 MB / 5000 张）
- 颜色直方图：`.fabric_cache/color_histograms.npy`（~1.5 MB / 5000 张）
- 文件变更（增删花型图）自动失效重建

## 评估结果（490 花型库, 10 测试照片）

| 指标 | 传统CV最优 | CLIP (本方案) |
|------|-----------|---------------|
| Top-1 | 10% | 10% |
| Top-3 | 10% | 10% |
| Top-5 | 10% | 40% |
| Top-10 | 10% | 50% |
| Top-100 | 50% | 80% |
| Mean Rank | ~180 | 45.9 |

## 开发历程

1. **v1**: 颜色直方图 + LBP + FFT → 75% (4 花型)
2. **v2**: + 颜色校准 + FFT Hann 窗 → 75%
3. **v3**: + ORB 特征点 + KD-tree 索引
4. **v4**: + Gabor/GLCM/自相关形状特征 → 40% (24 花型)
5. **v5**: + Z-score/Borda 融合/模板匹配/POC → 50% Top-100 (490 花型)
6. **CLIP+FAISS**: 深度学习方案 → 80% Top-100 (490 花型)
7. **CLIP+HSV+POC**: 混合检索 + 颜色融合 + 结构验证 → 当前方案

## License

MIT
