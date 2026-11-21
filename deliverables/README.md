# 交付材料导航

本目录集中放置课程报告、展示及修改 PPT 时需要的图素材。项目解释与复现步骤见 [文档导航](../docs/README.md)，完整实验数据见 [结果数据说明](../results/README.md)。

## report/：10 页英文报告

- [final_report.pdf](report/final_report.pdf)：可直接阅读的报告。
- [final_report.tex](report/final_report.tex)：LaTeX 源文件，`generated/` 中的数字和表格由真实结果自动生成。
- [author.tex](report/author.tex)：三位成员填写英文姓名和学号的位置。
- [build_manifest.json](report/build_manifest.json)：文档、数据和图表的构建校验记录。

## presentation/：12 分钟展示

- [slides.pdf](presentation/slides.pdf)：12 页展示。
- [slides.tex](presentation/slides.tex)：Beamer 源文件。
- [speaker_notes.md](presentation/speaker_notes.md)：英文讲稿、建议时间和三人分工。
- [QA_zh.md](presentation/QA_zh.md)：中文问答准备。

## figure_assets/：修改 PPT 的素材

- [index.html](figure_assets/index.html)：11 组图的离线预览页。
- [README.md](figure_assets/README.md)：图名、用途、对应展示页和指标说明。
- [png/](figure_assets/png/)：直接插入 PowerPoint、Keynote 或 WPS 的图片。
- [pdf/](figure_assets/pdf/)：相同内容的矢量图。

组员需要单独接收素材时，复制整个 `figure_assets/` 文件夹即可。正式报告的绘图来源仍是 `results/figures/`，素材文件是经过核对的副本。

## 修改与重新生成

在仓库根目录执行：

```bash
conda activate ppo
# 填写 deliverables/report/author.tex 后更新报告和展示：
python -m hvac_ppo report
# 更新原始图后同步素材副本：
python scripts/export_figure_assets.py
```

生成报告需要 TeX Live。构建命令同时更新中文结果 HTML、讲稿和问答；这些生成文件的长期修改应写进 `hvac_ppo/reporting.py`，避免下次构建覆盖。图素材的说明由 `scripts/export_figure_assets.py` 生成。
