# Fabric Pattern Matcher

面料花型智能匹配工具 — 拍一张照片，在花型库中检索最相似的花型。

## 技术架构

```
照片上传 → CLIP ViT-B/32 提取嵌入 → FAISS 向量检索 → 排序输出
```

- **CLIP** (OpenAI) — 预训练视觉模型，提取 512 维语义嵌入，光照/角度/尺度不变
- **FAISS** (Meta) — 向量索引，毫秒级检索，支持 10 万+ 花型库
- **Gradio** — Web UI，拖拽上传即用

每次检索耗时 ~50ms（不含模型首次加载）。

## 快速开始

```bash
# 安装依赖
pip install open-clip-torch faiss-cpu gradio pillow

# 启动 GUI
python app.py --fabric-dir dir/fabric

# 打开浏览器 http://127.0.0.1:7860
```

首次运行会下载 CLIP 模型（~300MB），之后使用本地缓存。

## 目录结构

```
fabric/
├── app.py                  # Gradio GUI + CLIP + FAISS 检索主程序
├── dir/
│   ├── fabric/             # 花型库（PNG/JPG，任意数量）
│   └── photo/              # 测试照片（10 张，供评估用）
├── matcher_v3/             # 传统 CV 方案（v1-v5 迭代记录）
│   ├── features.py         # 手工特征提取（LBP/Gabor/FFT/GLCM/形状）
│   ├── acorr_fingerprint.py # 自相关结构指纹
│   ├── matcher.py          # 多策略匹配器
│   └── clip_matcher.py     # ResNet50 方案（备选）
├── analyze_fabric.py       # v1 分析脚本
├── eval_clip.py            # CLIP 评估脚本
└── .fabric_cache/          # 嵌入缓存（自动生成）
```

## GUI 功能

- 拖拽/点击上传照片
- 置信度分级显示（极高/高/中/较低/低）
- 相似度进度条
- 清空/重新检索
- 返回数量可调

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

## License

MIT
