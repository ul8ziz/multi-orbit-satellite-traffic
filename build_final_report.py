# -*- coding: utf-8 -*-
"""Генерация итогового научного отчёта (DOCX) по результатам конвейера проекта."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

from src.reports.academic_ru_content import (
    explain_figure_blocking,
    explain_figure_corr_heat,
    explain_figure_forecast,
    explain_figure_heatmap,
    explain_figure_histogram,
    explain_figure_timeseries,
    paragraphs_abstract,
    paragraphs_conclusion,
    paragraphs_data,
    paragraphs_discussion,
    paragraphs_empirical_model_validation,
    paragraphs_forecast,
    paragraphs_integration,
    paragraphs_intro,
    paragraphs_limitations,
    paragraphs_math_model,
    paragraphs_model_results,
    paragraphs_processing,
    paragraphs_recommendations,
    paragraphs_spatial,
    paragraphs_statistics,
)


def _ensure_project_path(root: Path) -> None:
    r = str(root.resolve())
    if r not in sys.path:
        sys.path.insert(0, r)


def _save_document(doc: Document, out_docx: Path) -> Path:
    """Сохраняет DOCX во временный файл и атомарно подменяет цель; при блокировке — *_polished.docx."""
    out_docx.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_docx.with_name(out_docx.stem + ".tmp_write.docx")
    if tmp.is_file():
        tmp.unlink()
    doc.save(str(tmp))
    try:
        os.replace(str(tmp), str(out_docx))
        return out_docx
    except OSError:
        fb = out_docx.with_name(out_docx.stem + "_polished.docx")
        if fb.is_file():
            fb.unlink()
        os.replace(str(tmp), str(fb))
        print(
            f"Предупреждение: не удалось перезаписать {out_docx} (файл занят). "
            f"Сохранено: {fb}",
            file=sys.stderr,
        )
        return fb


def _set_doc_defaults(doc: Document) -> None:
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)
    style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    style.paragraph_format.space_after = Pt(6)
    try:
        style.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    except Exception:
        pass


def _p(
    doc: Document,
    text: str,
    *,
    bold: bool = False,
    italic: bool = False,
    align: Any | None = None,
) -> None:
    para = doc.add_paragraph()
    if align is not None:
        para.alignment = align
    run = para.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)
    try:
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    except Exception:
        pass
    para.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE


def _eq(doc: Document, text: str) -> None:
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = para.add_run(text)
    run.italic = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)
    try:
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    except Exception:
        pass
    para.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE


def _heading(doc: Document, text: str, level: int) -> None:
    h = doc.add_heading(text, level=level)
    for r in h.runs:
        r.font.name = "Times New Roman"
        try:
            r._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        except Exception:
            pass
    h.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE


def _add_table_caption(doc: Document, num: int, title: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"Таблица {num} – {title}")
    r.italic = True
    r.font.size = Pt(12)
    r.font.name = "Times New Roman"


def _add_figure_caption(doc: Document, num: int, title: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"Рисунок {num} – {title}")
    r.italic = True
    r.font.size = Pt(12)
    r.font.name = "Times New Roman"


def _add_figure(doc: Document, path: Path, num: int, title: str, width_in: float = 5.8) -> bool:
    if not path.is_file():
        return False
    doc.add_picture(str(path), width=Inches(width_in))
    _add_figure_caption(doc, num, title)
    return True


def _df_to_table(doc: Document, df: pd.DataFrame, float_fmt: str = "{:.6g}") -> None:
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Table Grid"
    hdr = t.rows[0].cells
    for i, c in enumerate(df.columns):
        hdr[i].text = str(c)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row):
            if isinstance(v, float):
                cells[i].text = float_fmt.format(v)
            else:
                cells[i].text = str(v)


def _display_app_class(label: str) -> str:
    words = str(label).split()
    acronyms = {"db": "DB", "p2p": "P2P", "voip": "VoIP"}
    return " ".join(acronyms.get(w.casefold(), w.capitalize()) for w in words)


def _add_paragraphs(doc: Document, paragraphs: list[str]) -> None:
    for t in paragraphs:
        _p(doc, t)


def load_stats(root: Path) -> dict[str, Any]:
    root = Path(root)
    proc = root / "data" / "processed"
    metrics_path = proc / "stage_a_metrics.json"
    pq_path = proc / "cleaned_traffic.parquet"
    csv_path = proc / "cleaned_traffic.csv"

    if metrics_path.is_file():
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        tmin = pd.Timestamp(m["time_min"])
        tmax = pd.Timestamp(m["time_max"])
        n_records = int(m["n_records_cleaned"])
        n_users = int(m["n_unique_users"])
        n_app_classes = int(m.get("n_unique_app_class", 0))
        n_hourly = int(m.get("n_hourly_buckets", m.get("n_hourly_aggregates", 0)))
        lam_mean = float(m["lambda_t"]["mean"])
        lam_peak = float(m["lambda_t"]["peak"])
    elif pq_path.is_file():
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(pq_path)
        n_records = pf.metadata.num_rows
        tcol = pd.read_parquet(pq_path, columns=["start_ts"])
        tmin = pd.to_datetime(tcol["start_ts"]).min()
        tmax = pd.to_datetime(tcol["start_ts"]).max()
        ucol = pd.read_parquet(pq_path, columns=["MASKED_MSISDN"])
        n_users = int(ucol["MASKED_MSISDN"].nunique())
        from src.analysis.slices import normalize_app_class

        app_col = pd.read_parquet(pq_path, columns=["APP_CLASS"])
        n_app_classes = int(app_col["APP_CLASS"].map(normalize_app_class).nunique())
        hourly_pq = proc / "hourly_slice_totals.parquet"
        if hourly_pq.is_file():
            h = pd.read_parquet(hourly_pq)
            h["date"] = pd.to_datetime(h["date"]).dt.normalize()
            buckets = h.groupby(["date", "hour"], as_index=False)["total_mb"].sum()
            lam_t = buckets["total_mb"] / 3600.0
            lam_mean = float(lam_t.mean())
            lam_peak = float(lam_t.max())
            n_hourly = len(buckets)
        else:
            lam_mean, lam_peak, n_hourly = 0.0, 0.0, 0
    else:
        cleaned = pd.read_csv(csv_path)
        cleaned["start_ts"] = pd.to_datetime(cleaned["start_ts"])
        tmin = cleaned["start_ts"].min()
        tmax = cleaned["start_ts"].max()
        n_records = len(cleaned)
        n_users = int(cleaned["MASKED_MSISDN"].nunique())
        from src.analysis.slices import normalize_app_class

        n_app_classes = int(cleaned["APP_CLASS"].map(normalize_app_class).nunique())
        hourly_buckets = cleaned.groupby(["date", "hour"], as_index=False)["total_mb"].sum()
        lam_t = hourly_buckets["total_mb"] / 3600.0
        lam_mean = float(lam_t.mean())
        lam_peak = float(lam_t.max())
        n_hourly = len(hourly_buckets)

    slice_stats_p = root / "outputs/tables/slice_descriptive_stats.csv"
    slice_stats = pd.read_csv(slice_stats_p)

    burst = pd.read_csv(root / "outputs/tables/burstiness_papr.csv")

    corr_p = root / "outputs/tables/slice_correlation_matrix.csv"
    corr = pd.read_csv(corr_p)

    fc_path = root / "outputs/tables/forecast_model_comparison.csv"
    fc = pd.read_csv(fc_path) if fc_path.is_file() else pd.DataFrame()

    markov_m = pd.read_csv(root / "outputs/markov/metrics.csv")
    scen = pd.read_csv(root / "outputs/integration/scenario_metrics.csv")
    stat_dist = pd.read_csv(root / "outputs/markov/stationary_distribution.csv")

    top_states = stat_dist.nlargest(10, "pi")[
        ["state_id", "n", "m", "pi"]
    ].reset_index(drop=True)

    return {
        "n_records": n_records,
        "n_users": n_users,
        "n_app_classes": n_app_classes,
        "tmin": tmin,
        "tmax": tmax,
        "n_days": (tmax.normalize() - tmin.normalize()).days + 1,
        "slice_stats": slice_stats,
        "burst": burst,
        "corr": corr,
        "forecast_cmp": fc,
        "markov_m": markov_m.iloc[0].to_dict(),
        "scenario": scen,
        "top_states": top_states,
        "lam_mean": lam_mean,
        "lam_peak": lam_peak,
        "n_hourly": n_hourly,
        "alpha_used": float(scen["alpha"].iloc[0]),
    }


def build_report(root: Path, out_docx: Path) -> Path:
    _ensure_project_path(root)
    stats = load_stats(root)
    doc = Document()
    _set_doc_defaults(doc)

    student = os.environ.get("REPORT_STUDENT_NAME", "Фамилия И. О.")
    university = os.environ.get("REPORT_UNIVERSITY", "Название образовательной организации")
    course = os.environ.get("REPORT_COURSE", "Учебная дисциплина / направление подготовки")
    year = os.environ.get("REPORT_YEAR", "2026")

    for _ in range(3):
        doc.add_paragraph()
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run(
        "Анализ пользовательского трафика и имитационное моделирование "
        "многоуровневой спутниковой сети LEO/MEO на основе марковской модели Леонтьевой"
    )
    r.bold = True
    r.font.size = Pt(14)
    r.font.name = "Times New Roman"
    doc.add_paragraph()
    for line in (student, university, course, year):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        rr = p.add_run(line)
        rr.font.size = Pt(12)
        rr.font.name = "Times New Roman"
    doc.add_page_break()

    _heading(doc, "Аннотация", 1)
    _add_paragraphs(doc, paragraphs_abstract(stats))
    doc.add_page_break()

    _heading(doc, "Введение", 1)
    _add_paragraphs(doc, paragraphs_intro())

    _heading(doc, "Описание исходных данных", 1)
    _add_paragraphs(doc, paragraphs_data(stats))

    _heading(doc, "Обработка данных и отображение на сетевые срезы", 1)
    _add_paragraphs(doc, paragraphs_processing())
    _add_table_caption(doc, 1, "Соответствие классов приложений срезам 3GPP")
    map_path = root / "outputs" / "reports" / "app_class_mapping_report.csv"
    if map_path.is_file():
        map_df = pd.read_csv(map_path)
    else:
        from src.analysis.slices import app_class_mapping_table

        map_df = app_class_mapping_table()
    map_df = map_df.rename(columns={"canonical_APP_CLASS": "Класс приложения", "slice_id": "Срез 3GPP"})
    map_df["Класс приложения"] = map_df["Класс приложения"].map(_display_app_class)
    _df_to_table(doc, map_df)
    doc.add_paragraph()
    _p(
        doc,
        f"Таблица задаёт детерминированное отображение для {stats['n_app_classes']} нормализованных классов, "
        "фактически присутствующих в очищенных данных: каждому эталонному названию класса соответствует ровно один срез. "
        "Это упрощение не учитывает смешанных приложений, но "
        "согласуется с инженерной практикой классификации QoS.",
    )

    fig_num = 1
    _heading(doc, "Пространственно-временное моделирование спроса", 1)
    _add_paragraphs(doc, paragraphs_spatial())
    hp = root / "outputs" / "figures" / "heatmap_hour_slice.png"
    if _add_figure(
        doc,
        hp,
        fig_num,
        "Тепловая карта средних почасовых объёмов по срезам (усреднение по календарным дням)",
    ):
        _p(doc, explain_figure_heatmap(fig_num))
        fig_num += 1

    _heading(doc, "Статистический анализ по срезам", 1)
    _add_paragraphs(doc, paragraphs_statistics(stats))
    _add_table_caption(doc, 2, "Выборочные среднее, дисперсия, СКО и пик-фактор по суточному почасовому профилю")
    _df_to_table(doc, stats["slice_stats"])
    doc.add_paragraph()
    _p(
        doc,
        "Интерпретация столбцов: mean и variance относятся к двадцатичетырёхчасовому вектору усреднённых по дням "
        "почасовых сумм; papr характеризует отношение максимума этого вектора к среднему; peak_hour_avg и "
        "off_peak_hour_avg разделяют часы выше и ниже среднего уровня профиля внутри среза.",
    )
    _add_table_caption(doc, 3, "Пик-фактор и показатель всплесковости (квадрат коэффициента вариации профиля)")
    _df_to_table(doc, stats["burst"])
    doc.add_paragraph()
    _p(
        doc,
        "Сопоставление PAPR и burstiness_cv2 позволяет отделить сценарии с одним выраженным пиком от сценариев с "
        "общей высокой вариабельностью без доминирования одного часа.",
    )
    _add_table_caption(doc, 4, "Корреляции Пирсона между суммарными почасовыми рядами срезов")
    _df_to_table(doc, stats["corr"])
    doc.add_paragraph()

    ts = root / "outputs" / "figures" / "timeseries_hourly_by_slice.png"
    if _add_figure(doc, ts, fig_num, "Почасовые суммы трафика по сетевым срезам"):
        _p(doc, explain_figure_timeseries(fig_num))
        fig_num += 1
    hg = root / "outputs" / "figures" / "histogram_total_mb.png"
    if _add_figure(doc, hg, fig_num, "Распределение объёма сессии (total_mb) после очистки"):
        _p(doc, explain_figure_histogram(fig_num))
        fig_num += 1
    ch = root / "outputs" / "figures" / "correlation_heatmap.png"
    if _add_figure(doc, ch, fig_num, "Визуализация корреляционной матрицы между срезами"):
        _p(doc, explain_figure_corr_heat(fig_num))
        fig_num += 1

    _heading(doc, "Прогнозирование трафика", 1)
    _heading(doc, "4.3. Результаты прогнозирования — графики временных рядов (по слайсам)", 2)
    _add_paragraphs(doc, paragraphs_forecast(stats))
    _add_table_caption(doc, 5, "Сравнение моделей прогнозирования на тестовом интервале")
    _df_to_table(doc, stats["forecast_cmp"])
    doc.add_paragraph()
    _p(
        doc,
        "Столбцы rmse и mae отражают среднеквадратичную и среднюю абсолютную ошибку на тесте; spike_mae и "
        "spike_rmse усредняют ошибку только в точках, где факт превышает квантиль обучающего ряда; n_spikes — "
        "число таких точек; n_test — длина тестового горизонта.",
    )
    for fn, ttl, sru in [
        ("forecast_vs_actual_URLLC.png", "Сравнение прогноза и факта: URLLC", "URLLC"),
        ("forecast_vs_actual_eMBB.png", "Сравнение прогноза и факта: eMBB", "eMBB"),
        ("forecast_vs_actual_mMTC.png", "Сравнение прогноза и факта: mMTC", "mMTC"),
    ]:
        fp = root / "outputs" / "figures" / fn
        if _add_figure(doc, fp, fig_num, ttl):
            _p(doc, explain_figure_forecast(fig_num, sru))
            fig_num += 1

    _heading(doc, "Математическая модель Леонтьевой (точная CTMC)", 1)
    _add_paragraphs(doc, paragraphs_math_model(stats["markov_m"]))
    _p(
        doc,
        "Матрица инфинитезимальных интенсивностей Q определяется на конечном пространстве состояний размерности "
        "(N+1)(M+1). Для любых i≠j элемент Q_ij равен суммарной интенсивности всех переходов из состояния i в j; "
        "диагональ Q_ii выбирается так, чтобы выполнялось ∑_j Q_ij=0. Ниже записаны ключевые соотношения в компактной "
        "форме.",
    )
    _eq(doc, "μ(n) = min(n, C_LEO) · μ_LEO + max(0, n − C_LEO) · μ_MEO")
    _eq(doc, "π Q = 0,    ∑_k π_k = 1")

    _heading(doc, "Результаты моделирования CTMC (базовые параметры Stage B)", 1)
    _add_paragraphs(doc, paragraphs_model_results(stats["markov_m"]))
    _add_table_caption(doc, 6, "Фрагмент стационарного распределения (10 наиболее вероятных состояний)")
    _df_to_table(doc, stats["top_states"])
    doc.add_paragraph()
    _p(
        doc,
        "Вероятности π быстро убывают по мере удаления от области малых n и m: это качественно согласуется с "
        "умеренной интенсивностью поступления λ в базовом сценарии Stage B и с тем, что система преимущественно "
        "находится в конфигурациях с низкой одновременной загрузкой.",
    )

    _heading(doc, "Интеграция эмпирических данных и марковской модели", 1)
    _add_paragraphs(doc, paragraphs_integration(stats))
    _add_table_caption(doc, 7, "Показатели модели для сценариев mean_load и peak_load")
    _df_to_table(doc, stats["scenario"])
    doc.add_paragraph()
    _p(
        doc,
        "Сравнение строк таблицы показывает, как при фиксированных N, M, γ, C_LEO, μ_LEO, μ_MEO увеличение λ "
        "с типичного почасового уровня до пикового наблюдаемого в данных усиливает среднюю занятость и вероятности "
        "блокировок, оставаясь в рамках одной и той же структуры Q.",
    )
    bp = root / "outputs" / "integration" / "blocking_vs_scenario.png"
    if _add_figure(
        doc,
        bp,
        fig_num,
        "Стационарная вероятность P(n=N) для сценариев средней и пиковой нагрузки (данные)",
    ):
        _p(doc, explain_figure_blocking(fig_num))
        fig_num += 1

    _heading(doc, "Сопоставление эмпирических данных и модели (Stage A и Stage B)", 1)
    _add_paragraphs(doc, paragraphs_empirical_model_validation(stats))

    _heading(doc, "Обсуждение", 1)
    _add_paragraphs(doc, paragraphs_discussion(stats))

    _heading(doc, "Практические рекомендации", 1)
    _add_paragraphs(doc, paragraphs_recommendations())

    _heading(doc, "Ограничения", 1)
    _add_paragraphs(doc, paragraphs_limitations())

    _heading(doc, "Заключение", 1)
    _add_paragraphs(doc, paragraphs_conclusion())

    return _save_document(doc, out_docx)


def main() -> int:
    root = Path(__file__).resolve().parent
    _ensure_project_path(root)
    out = root / "outputs" / "reports" / "final_report.docx"
    try:
        build_report(root, out)
    except FileNotFoundError as e:
        print(f"Ошибка: не найдены входные данные: {e}", file=sys.stderr)
        print("Выполните: python run_pipeline.py", file=sys.stderr)
        return 1
    print("Final academic polish completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
