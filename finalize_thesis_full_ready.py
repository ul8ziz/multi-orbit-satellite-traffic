# -*- coding: utf-8 -*-
"""Finalize thesis DOCX with final calibration figures and analysis."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DOC = PROJECT_ROOT / "final_thesis_chapter_improved_polished.docx"
OUTPUT_DOC = PROJECT_ROOT / "final_thesis_FULL_READY.docx"
ANALYSIS_TXT = PROJECT_ROOT / "outputs" / "final_figures" / "final_figures_analysis.txt"
FIG_DIR = PROJECT_ROOT / "outputs" / "final_figures"

FINAL_ALPHA = [100.0, 300.0, 1000.0]


def _font_run(run, *, size: int = 12, bold: bool = False, italic: bool = False) -> None:
    run.bold = bold
    run.italic = italic
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    try:
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    except Exception:
        pass


def _set_paragraph_text(par, text: str, *, size: int = 12, bold: bool = False, italic: bool = False) -> None:
    par.clear()
    run = par.add_run(text)
    _font_run(run, size=size, bold=bold, italic=italic)
    par.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE


def _load_calibration_selected() -> pd.DataFrame:
    files = [
        PROJECT_ROOT / "outputs" / "calibration_results.csv",
        PROJECT_ROOT / "outputs" / "calibration_results_extended.csv",
        PROJECT_ROOT / "outputs" / "calibration_results_extended2.csv",
    ]
    parts = [pd.read_csv(p) for p in files if p.is_file()]
    if not parts:
        raise FileNotFoundError("Calibration CSV files not found in outputs/.")
    df = pd.concat(parts, ignore_index=True)
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    df = df.dropna(subset=["alpha"]).sort_values("alpha").drop_duplicates("alpha", keep="last")
    sel = df[df["alpha"].isin(FINAL_ALPHA)].copy().sort_values("alpha")
    if len(sel) != 3:
        raise ValueError("Expected calibration rows for alpha 100/300/1000.")
    return sel


def _parse_analysis_blocks(path: Path) -> dict[str, dict[str, str]]:
    txt = path.read_text(encoding="utf-8")
    blocks: dict[str, dict[str, str]] = {}
    parts = re.split(r"\nРисунок:\s*", "\n" + txt)
    for chunk in parts:
        c = chunk.strip()
        if not c:
            continue
        lines = c.splitlines()
        fname = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        sec = {
            "Описание": "",
            "Интерпретация": "",
            "Вывод": "",
            "Инженерный смысл": "",
        }
        for key in sec:
            m = re.search(rf"1\.\s*Описание|2\.\s*Интерпретация|3\.\s*Вывод|4\.\s*Инженерный смысл", body)
            if m:
                pass
        # deterministic parse by section numbers
        patt = (
            r"1\.\s*Описание\s*(.*?)\s*"
            r"2\.\s*Интерпретация\s*(.*?)\s*"
            r"3\.\s*Вывод\s*(.*?)\s*"
            r"4\.\s*Инженерный смысл\s*(.*)$"
        )
        mm = re.search(patt, body, flags=re.S)
        if mm:
            sec["Описание"] = mm.group(1).strip()
            sec["Интерпретация"] = mm.group(2).strip()
            sec["Вывод"] = mm.group(3).strip()
            sec["Инженерный смысл"] = mm.group(4).strip()
        blocks[fname] = sec
    return blocks


def _rewrite_tables(doc: Document, sel: pd.DataFrame) -> None:
    # table 4: alpha sweep summary -> keep only final scenarios
    if len(doc.tables) > 4:
        t = doc.tables[4]
        rows = [
            [
                f"{int(r['alpha'])}",
                f"{r['P_block']:.6g}",
                f"{r['E[n]']:.6g}",
                f"{r['E[m]']:.6g}",
                f"{r['util_LEO']:.6g}",
                f"{r['util_MEO']:.6g}",
                "False",
            ]
            for _, r in sel.iterrows()
        ]
        # ensure enough rows
        while len(t.rows) < 1 + len(rows):
            t.add_row()
        for i, row_vals in enumerate(rows, start=1):
            for j, v in enumerate(row_vals):
                t.rows[i].cells[j].text = str(v)
        for i in range(1 + len(rows), len(t.rows)):
            for j in range(len(t.rows[i].cells)):
                t.rows[i].cells[j].text = ""

    # table 5: recommended scenarios table -> overwrite rows with final 100/300/1000
    if len(doc.tables) > 5:
        t = doc.tables[5]
        cols = [c.text for c in t.rows[0].cells]
        rec_labels = ["recommended_low", "recommended_medium", "recommended_high"]
        while len(t.rows) < 4:
            t.add_row()
        for i, ((_, r), rec) in enumerate(zip(sel.iterrows(), rec_labels), start=1):
            vals = {
                "stage": "final_recommended",
                "alpha": f"{int(r['alpha'])}",
                "N": "5",
                "M": "4",
                "C_LEO": "2",
                "mu_LEO": "3.0",
                "mu_MEO": "2.0",
                "gamma": "1.0",
                "scenario": "mean_load",
                "lambda": f"{r['lambda_mean']:.6g}",
                "lambda_mean_implied": f"{r['lambda_mean']:.6g}",
                "lambda_peak_implied": f"{r['lambda_peak']:.6g}",
                "P(n=N)": f"{r['P_block']:.6g}",
                "P(m=M)": f"{r['P(m=M)']:.6g}",
                "E[n]": f"{r['E[n]']:.6g}",
                "E[m]": f"{r['E[m]']:.6g}",
                "util_LEO": f"{r['util_LEO']:.6g}",
                "util_MEO": f"{r['util_MEO']:.6g}",
                "saturation_flag": "False",
                "trivial_underload_flag": "False",
                "error": "",
                "recommendation": rec,
            }
            for j, c in enumerate(cols):
                t.rows[i].cells[j].text = vals.get(c, "")
        for i in range(4, len(t.rows)):
            for j in range(len(t.rows[i].cells)):
                t.rows[i].cells[j].text = ""


def _update_textual_values(doc: Document, sel: pd.DataFrame) -> None:
    r100, r300, r1000 = sel.iloc[0], sel.iloc[1], sel.iloc[2]
    for p in doc.paragraphs:
        t = p.text.strip()
        if not t:
            continue
        if t.startswith("3) Почему «сырой» результат"):
            _set_paragraph_text(
                p,
                "3) Почему исходный немасштабированный прогон выглядел насыщенным:",
            )
        elif t.startswith("На реальных данных при alpha=1.0"):
            _set_paragraph_text(
                p,
                "На реальных данных без согласования масштаба входная интенсивность по порядку "
                "величины существенно превосходила сервисную мощность выбранных параметров CTMC, "
                "что приводило к переоценке насыщения. Финальная калибровка с alpha=100/300/1000 "
                "устраняет эту несогласованность и задаёт интерпретируемые эксплуатационные режимы.",
            )
        # Replace old recommended lines inside calibration summary text
        if t.startswith("recommended_low:"):
            _set_paragraph_text(
                p,
                f"recommended_low: alpha=100, N=5, M=4, C_LEO=2, mu_LEO=3.0, mu_MEO=2.0, gamma=1.0, "
                f"P(n=N)={r100['P_block']:.4f}, util_LEO={r100['util_LEO']:.4f}",
            )
        elif t.startswith("recommended_medium:"):
            _set_paragraph_text(
                p,
                f"recommended_medium: alpha=300, N=5, M=4, C_LEO=2, mu_LEO=3.0, mu_MEO=2.0, gamma=1.0, "
                f"P(n=N)={r300['P_block']:.4f}, util_LEO={r300['util_LEO']:.4f}",
            )
        elif t.startswith("recommended_high:"):
            _set_paragraph_text(
                p,
                f"recommended_high: alpha=1000, N=5, M=4, C_LEO=2, mu_LEO=3.0, mu_MEO=2.0, gamma=1.0, "
                f"P(n=N)={r1000['P_block']:.4f}, util_LEO={r1000['util_LEO']:.4f}",
            )
        elif t.startswith("Описание параметров и метрик.") and "recommended_low" not in t:
            # leave; handled below by heading context
            pass

    # Update scenario blocks by heading context
    current = ""
    for p in doc.paragraphs:
        t = p.text.strip()
        if t.startswith("2.5.1"):
            current = "low"
        elif t.startswith("2.5.2"):
            current = "medium"
        elif t.startswith("2.5.3"):
            current = "high"
        elif t.startswith("2.5.4"):
            current = "cmp"
        if t.startswith("Описание параметров и метрик."):
            if current == "low":
                _set_paragraph_text(
                    p,
                    "Описание параметров и метрик. "
                    f"alpha=100; N=5; M=4; C_LEO=2; mu_LEO=3.0; mu_MEO=2.0; gamma=1.0; "
                    f"lambda={r100['lambda_mean']:.6g}; P(n=N)={r100['P_block']:.6g}; "
                    f"P(m=M)={r100['P(m=M)']:.6g}; E[n]={r100['E[n]']:.6g}; E[m]={r100['E[m]']:.6g}; "
                    f"util_LEO={r100['util_LEO']:.6g}; util_MEO={r100['util_MEO']:.6g}.",
                )
            elif current == "medium":
                _set_paragraph_text(
                    p,
                    "Описание параметров и метрик. "
                    f"alpha=300; N=5; M=4; C_LEO=2; mu_LEO=3.0; mu_MEO=2.0; gamma=1.0; "
                    f"lambda={r300['lambda_mean']:.6g}; P(n=N)={r300['P_block']:.6g}; "
                    f"P(m=M)={r300['P(m=M)']:.6g}; E[n]={r300['E[n]']:.6g}; E[m]={r300['E[m]']:.6g}; "
                    f"util_LEO={r300['util_LEO']:.6g}; util_MEO={r300['util_MEO']:.6g}.",
                )
            elif current == "high":
                _set_paragraph_text(
                    p,
                    "Описание параметров и метрик. "
                    f"alpha=1000; N=5; M=4; C_LEO=2; mu_LEO=3.0; mu_MEO=2.0; gamma=1.0; "
                    f"lambda={r1000['lambda_mean']:.6g}; P(n=N)={r1000['P_block']:.6g}; "
                    f"P(m=M)={r1000['P(m=M)']:.6g}; E[n]={r1000['E[n]']:.6g}; E[m]={r1000['E[m]']:.6g}; "
                    f"util_LEO={r1000['util_LEO']:.6g}; util_MEO={r1000['util_MEO']:.6g}.",
                )
        if t.startswith("Количественная интерпретация."):
            if current == "low":
                _set_paragraph_text(
                    p,
                    "Количественная интерпретация. "
                    f"При α=100 получено P(n=N)={r100['P_block']:.6g} и util_LEO={r100['util_LEO']:.6g}; "
                    "это соответствует режиму «недогруженный» и подтверждает наличие эксплуатационного запаса.",
                )
            elif current == "medium":
                _set_paragraph_text(
                    p,
                    "Количественная интерпретация. "
                    f"При α=300 получено P(n=N)={r300['P_block']:.6g} и util_LEO={r300['util_LEO']:.6g}; "
                    "это соответствует режиму «сбалансированный» и определяет рабочую точку recommended_medium.",
                )
            elif current == "high":
                _set_paragraph_text(
                    p,
                    "Количественная интерпретация. "
                    f"При α=1000 получено P(n=N)={r1000['P_block']:.6g} и util_LEO={r1000['util_LEO']:.6g}; "
                    "это соответствует высоконагруженному режиму с повышенным риском деградации QoS.",
                )

        if t.startswith("Сравнение трёх рекомендованных сценариев"):
            _set_paragraph_text(
                p,
                "Сравнение трёх рекомендованных сценариев подтверждает монотонный рост нагрузки: "
                f"P(n=N) увеличивается от {r100['P_block']:.6g} (recommended_low, alpha=100) "
                f"до {r300['P_block']:.6g} (recommended_medium, alpha=300) "
                f"и {r1000['P_block']:.6g} (recommended_high, alpha=1000).",
            )
        if t.startswith("Утилизация LEO демонстрирует согласованную картину"):
            _set_paragraph_text(
                p,
                "Утилизация LEO возрастает от "
                f"{r100['util_LEO']:.6g} (alpha=100) к {r300['util_LEO']:.6g} (alpha=300) "
                f"и {r1000['util_LEO']:.6g} (alpha=1000), что отражает переход от недогруза к "
                "высоконагруженному режиму. Точка alpha=300 находится в области управляемого "
                "компромисса между эффективностью и вероятностью отказов.",
            )
        if t.startswith("Итоговый научный вывод по разделу 2.5."):
            _set_paragraph_text(
                p,
                "Итоговый научный вывод по разделу 2.5. В терминах совместной оптимизации "
                "блокировки и утилизации оптимальным эксплуатационным режимом является "
                "recommended_medium (alpha=300): при P(n=N)=0.0328 система остаётся устойчивой, "
                "обеспечивая при этом высокую util_LEO=0.6597 без перехода к зоне тяжёлой перегрузки.",
            )
        if t.startswith("Инженерная интерпретация."):
            _set_paragraph_text(
                p,
                "Инженерная интерпретация. Для реальных многоуровневых спутниковых систем LEO/MEO "
                "режим alpha=300 предпочтителен как базовая рабочая точка: он поддерживает "
                "осмысленную загрузку ресурсов, ограничивает рост орбиты ретраев и уменьшает "
                "риск деградации QoS по сравнению с высоконагруженным сценарием alpha=1000.",
            )

        if t.startswith("Практический смысл для операторов многоуровневой спутниковой сети"):
            _set_paragraph_text(
                p,
                "Практический смысл для операторов многоуровневой спутниковой сети заключается "
                "в использовании трёх калиброванных режимов: recommended_low (alpha=100) как "
                "защитного режима, recommended_medium (alpha=300) как целевого эксплуатационного "
                "режима и recommended_high (alpha=1000) как стресс-сценария для верификации "
                "устойчивости и резервирования.",
            )


def _insert_figure_block_before(anchor_par, fig_path: Path, caption: str, block: dict[str, str]) -> None:
    p_img = anchor_par.insert_paragraph_before()
    run = p_img.add_run()
    run.add_picture(str(fig_path), width=Inches(5.8))
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER

    p_cap = anchor_par.insert_paragraph_before()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p_cap.add_run(caption)
    _font_run(r, size=12, italic=True)

    for key in ("Описание", "Интерпретация", "Вывод", "Инженерный смысл"):
        p = anchor_par.insert_paragraph_before()
        r = p.add_run(f"{key}. {block.get(key, '').strip()}")
        _font_run(r, size=12)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE


def _insert_final_figures(doc: Document, analysis: dict[str, dict[str, str]]) -> None:
    para_25 = None
    para_26 = None
    for p in doc.paragraphs:
        t = p.text.strip()
        if para_25 is None and t.startswith("2.5 Анализ рекомендованных сценариев"):
            para_25 = p
        if para_26 is None and t.startswith("2.6 Итоговые выводы"):
            para_26 = p
    if para_25 is None or para_26 is None:
        raise RuntimeError("Could not locate section anchors 2.5/2.6 in document.")

    # Insert six final calibration figures in section 2.4 (before 2.5 heading)
    first_six = [
        ("blocking_vs_alpha_final.png", "Рисунок 2.5 – Вероятность блокировки P(n=N) в зависимости от alpha"),
        ("En_vs_alpha_final.png", "Рисунок 2.6 – Среднее число активных заявок E[n] в зависимости от alpha"),
        ("Em_vs_alpha_final.png", "Рисунок 2.7 – Средний размер орбиты ретраев E[m] в зависимости от alpha"),
        ("PmM_vs_alpha_final.png", "Рисунок 2.8 – Вероятность переполнения орбиты P(m=M) в зависимости от alpha"),
        ("utilization_vs_alpha_final.png", "Рисунок 2.9 – Утилизация ресурсов LEO и MEO в зависимости от alpha"),
        ("tradeoff_final.png", "Рисунок 2.10 – Компромисс между блокировкой и утилизацией LEO"),
    ]
    for fname, cap in first_six:
        fp = FIG_DIR / fname
        if not fp.is_file():
            continue
        _insert_figure_block_before(para_25, fp, cap, analysis.get(fname, {}))

    # Insert scenario comparison in section 2.5 (before 2.6 heading)
    fname = "scenario_comparison_final.png"
    fp = FIG_DIR / fname
    if fp.is_file():
        _insert_figure_block_before(
            para_26,
            fp,
            "Рисунок 2.11 – Сравнение финальных сценариев recommended_low / medium / high",
            analysis.get(fname, {}),
        )


def main() -> None:
    if not INPUT_DOC.is_file():
        raise FileNotFoundError(INPUT_DOC)
    if not ANALYSIS_TXT.is_file():
        raise FileNotFoundError(ANALYSIS_TXT)

    sel = _load_calibration_selected()
    analysis = _parse_analysis_blocks(ANALYSIS_TXT)

    doc = Document(str(INPUT_DOC))
    _rewrite_tables(doc, sel)
    _update_textual_values(doc, sel)
    _insert_final_figures(doc, analysis)

    doc.save(str(OUTPUT_DOC))
    print(OUTPUT_DOC.resolve())


if __name__ == "__main__":
    main()
