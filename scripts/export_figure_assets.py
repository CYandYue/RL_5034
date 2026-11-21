"""Copy existing figures into a portable, documented presentation asset folder.

Uses only the Python standard library. Run again after regenerating results/figures.
"""

from pathlib import Path
import hashlib
import html
import json
import shutil


ROOT = Path(__file__).resolve().parents[1]
FIGURES = (
    ("slides_learning", "展示用学习曲线", "PPT", "第 5 页", "七个配置的平均验证代价，纵轴为对数；越低越好。"),
    ("slides_clip", "Clip range 消融", "PPT", "第 7 页", "clip=0.1/0.2/0.3 对回报、越界率和电量的影响。"),
    ("slides_lambda", "GAE lambda 消融", "PPT", "第 8 页", "lambda=0.90/0.95/0.99 对回报、越界率和电量的影响。"),
    ("slides_epochs", "Epochs 消融", "PPT", "第 9 页", "epochs=5/10/20 对回报、越界率和电量的影响。"),
    ("slides_cold_trace", "冷天闭环轨迹", "PPT", "第 10 页", "默认 PPO seed 0 与恒温器的室温及累计电量；固定条件 test_cold_00。"),
    ("learning_curves", "完整与后期学习曲线", "报告", "第 6 页", "上排展示完整训练，下排放大 102,400 步后的变化；阴影为跨种子标准差。"),
    ("comfort_energy_tradeoff", "舒适度与电量权衡", "报告", "第 7 页", "横轴为七天电量，纵轴为越界率；含恒温器参考，未画关闭空调。"),
    ("update_diagnostics", "PPO 更新诊断", "报告", "第 8 页", "近似 KL 与裁剪比例；跨种子均值后进行五次更新移动平均。"),
    ("representative_trajectories", "三类天气闭环轨迹", "报告", "第 9 页", "室温、外温、动作与累计电量；默认 PPO seed 0 和恒温器使用相同条件。"),
    ("test_metrics", "完整物理指标比较", "补充", "中文解读", "三个参数分别比较 CVR、越界度时和电量；可用于拆分或扩展结果页。"),
    ("runtime", "训练耗时比较", "补充", "中文解读", "训练墙钟时间扣除周期评价；误差条为跨种子标准差。"),
)


def export_assets():
    source = ROOT / "results/figures"
    output = ROOT / "deliverables/figure_assets"
    missing = [str(source / f"{name}.{extension}") for name, *_ in FIGURES
               for extension in ("png", "pdf") if not (source / f"{name}.{extension}").is_file()]
    if missing:
        raise FileNotFoundError("Generate the source figures first:\n" + "\n".join(missing))
    for extension in ("png", "pdf"):
        (output / extension).mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "figure_count": len(FIGURES), "files": []}
    rows, cards = [], []
    for name, title, category, location, description in FIGURES:
        for extension in ("png", "pdf"):
            original = source / f"{name}.{extension}"
            destination = output / extension / original.name
            shutil.copyfile(original, destination)
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            if digest != hashlib.sha256(original.read_bytes()).hexdigest():
                raise RuntimeError(f"Asset copy differs from source: {destination}")
            manifest["files"].append({"source": str(original.relative_to(ROOT)),
                                      "path": str(destination.relative_to(output)),
                                      "sha256": digest, "bytes": destination.stat().st_size})
        rows.append(f"| {title} | `{name}` | {category}：{location} | {description} |")
        cards.append(f'''<article><p class="tag">{html.escape(category)} · {html.escape(location)}</p>
<h2>{html.escape(title)}</h2><a href="png/{name}.png"><img loading="lazy" src="png/{name}.png" alt="{html.escape(title)}"></a>
<p>{html.escape(description)}</p><p class="files"><a href="png/{name}.png">PNG 图片</a><a href="pdf/{name}.pdf">PDF 矢量图</a></p>
<code>{name}</code></article>''')
    readme = '''# 绘图素材：供修改 PPT / 报告使用

直接打开 **[index.html](index.html)** 预览全部图，再选择需要的格式。可以把整个 `deliverables/figure_assets/` 文件夹复制给组员，离线预览不依赖 Python、网络或训练模型。

- `png/`：11 张 PNG，最适合直接插入 PowerPoint、Keynote、WPS。优先使用 `slides_*`，文字和布局已经按展示场景调整。
- `pdf/`：对应的 11 张矢量 PDF，适合 LaTeX、打印和支持 PDF 的排版软件；放大更清晰。需要改变曲线、字体或误差条时，应从数据重新绘图。
- `manifest.json`：每个副本的原始来源、SHA-256 和文件大小；素材与正式报告图逐字节相同。

## 图名与用途

| 图 | 文件名（不含扩展名） | 现有位置 | 内容 |
| --- | --- | --- | --- |
''' + "\n".join(rows) + '''

## 使用时保留的说明

所有结果来自每配置 3 个训练种子、每种子 512,000 个交互步。最终测试先对每个模型的 30 个条件求均值，再跨 3 个训练种子求均值和样本标准差。误差条和学习曲线阴影表示训练种子的标准差；不是置信区间。

回报越高越好；越界率 CVR、越界度时、用电量越低越好。展示用学习曲线画的是 **负回报（代价，对数纵轴）**，因此越低越好。电量单位为 **kWh / 七天回合**，不是平均功率。更新诊断图已经做五次更新移动平均，其他图不要额外平滑后仍沿用原说明。

原始绘图入口在 `hvac_ppo/analysis.py`，数据在 `results/*.csv`；`results/README.md` 说明各表字段。此目录是供展示编辑使用的副本，正式报告仍引用 `results/figures/`。

## 更新素材

重新汇总绘图后，在仓库根目录执行：

```bash
python scripts/export_figure_assets.py
```

此命令仅复制已有图、核对副本哈希并更新说明和预览页，无需重跑训练或安装额外依赖。手工改图请另存文件，避免下次同步覆盖。
'''
    (output / "README.md").write_text(readme, encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    document = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>PPO 温控项目 · 图素材</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f1f4ef;color:#253b31;font:16px/1.7 system-ui,"PingFang SC",sans-serif}
main{max-width:1280px;margin:auto;padding:36px 24px}h1,h2{color:#20583f}h1{margin-bottom:8px}h2{margin:4px 0 16px;font-size:21px}
.intro{max-width:900px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px;margin-top:28px}
article{background:#fffefa;padding:22px;border:1px solid #dce4dd;border-radius:10px;min-width:0}
.tag{color:#65776a;font-size:13px;margin:0}img{display:block;width:100%;height:300px;object-fit:contain;background:white}
a{color:#216858}.files{display:flex;gap:24px}code{font-size:12px;overflow-wrap:anywhere;color:#65776a}
@media(max-width:760px){.grid{grid-template-columns:1fr}main{padding:24px 14px}img{height:auto}}
</style></head><body><main><h1>PPO 温控项目 · 图素材</h1>
<div class="intro"><p>11 组图，每组提供 PNG 与矢量 PDF。点击图片可查看原尺寸，使用“PNG 图片”链接选择插入 PPT 的文件。
优先使用标题标为 PPT 的图；其余图可作为补充。</p>
<p>误差条与阴影为三个训练种子的样本标准差。回报越高越好；“展示用学习曲线”绘制负回报，越低越好。
电量按七天回合统计。完整使用说明见 <a href="README.md">README</a>。</p></div>
<div class="grid">''' + "\n".join(cards) + "</div></main></body></html>\n"
    (output / "index.html").write_text(document, encoding="utf-8")
    print(f"Exported {len(FIGURES)} figures in PNG and PDF, with an offline gallery: {output}")


if __name__ == "__main__":
    export_assets()
