
from pathlib import Path

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, dash_table, Input, Output


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "evaluation_data.csv"

HORIZONS = ["1h", "2h", "3h"]

CLASS_ORDER = [
    "Không mưa",
    "Mưa nhẹ",
    "Mưa vừa",
    "Mưa lớn",
    "Mưa rất lớn"
]

SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]

PLOT_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "responsive": True
}


# ============================================================
# 2. LOAD DATA
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Chưa có file dữ liệu: {DATA_PATH}\n"
        "Hãy chạy cell export trong notebook để tạo "
        "data/evaluation_data.csv."
    )

df = pd.read_csv(DATA_PATH)

required = {
    "Horizon",
    "issue_time_utc",
    "valid_time",
    "rain_eval",
    "nowcast",
    "error",
    "absolute_error",
    "obs_class",
    "forecast_class",
    "event_type",
    "season",
    "hour_local"
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"File export thiếu các cột: {sorted(missing)}"
    )


# Convert datetime
for col in ["issue_time_utc", "valid_time"]:
    df[col] = pd.to_datetime(
        df[col],
        utc=True,
        errors="coerce"
    )


# Convert numeric
for col in [
    "rain_eval",
    "nowcast",
    "error",
    "absolute_error",
    "hour_local"
]:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# Remove invalid evaluation pairs
df = df.dropna(
    subset=[
        "valid_time",
        "rain_eval",
        "nowcast",
        "Horizon"
    ]
).copy()


# Convert categorical columns
for col in [
    "Horizon",
    "obs_class",
    "forecast_class",
    "event_type",
    "season"
]:
    df[col] = df[col].astype(str)


# Local time: UTC+7
df["valid_time_local"] = (
    df["valid_time"] + pd.Timedelta(hours=7)
)

df["date_local"] = (
    df["valid_time_local"].dt.date.astype(str)
)


MIN_DATE = df["valid_time_local"].min().date()
MAX_DATE = df["valid_time_local"].max().date()


# ============================================================
# 3. DASH APP
# ============================================================

app = Dash(
    __name__,
    title="WeatherPlus Nowcast | Verification"
)


app.index_string = """<!DOCTYPE html>
<html lang="vi">
<head>
    {%metas%}
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{%title%}</title>
    {%favicon%}
    {%css%}

    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>

    <link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap"
          rel="stylesheet">
</head>

<body>
    {%app_entry%}

    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>"""


# ============================================================
# 4. REUSABLE COMPONENTS
# ============================================================

def card(label, value_id, hint, accent):

    return html.Div(
        [
            html.Div(
                label,
                className="kpi-label"
            ),

            html.Div(
                id=value_id,
                className="kpi-value"
            ),

            html.Div(
                hint,
                className="kpi-hint"
            )
        ],
        className=f"kpi-card {accent}"
    )


def panel(
    title,
    subtitle,
    graph_id,
    class_name="panel",
    graph_height=350
):

    return html.Section(
        [
            html.Div(
                [
                    html.H3(title),

                    html.P(
                        subtitle,
                        className="panel-subtitle"
                    )
                ],
                className="panel-heading"
            ),

            dcc.Graph(
                id=graph_id,
                config=PLOT_CONFIG,
                className="chart",
                style={
                    "height": f"{graph_height}px",
                    "width": "100%"
                }
            )
        ],
        className=class_name
    )


# ============================================================
# 5. SIDEBAR
# ============================================================

sidebar = html.Aside(
    [
        html.Div(
            [
                html.Div(
                    "W",
                    className="brand-mark"
                ),

                html.Div(
                    [
                        html.Div(
                            "WEATHERPLUS",
                            className="brand-name"
                        ),

                        html.Div(
                            "NOWCAST VERIFICATION",
                            className="brand-sub"
                        )
                    ]
                )
            ],
            className="brand"
        ),

        html.Div(
            "WORKSPACE",
            className="side-label"
        ),

        html.A(
            [
                html.Span(
                    "▦",
                    className="nav-icon"
                ),
                "Tổng quan đánh giá"
            ],
            href="#overview",
            className="nav-item active"
        ),

        html.A(
            [
                html.Span(
                    "◷",
                    className="nav-icon"
                ),
                "Phân tích theo thời gian"
            ],
            href="#time-analysis",
            className="nav-item"
        ),

        html.A(
            [
                html.Span(
                    "▤",
                    className="nav-icon"
                ),
                "Đánh giá cường độ"
            ],
            href="#intensity-analysis",
            className="nav-item"
        ),

        html.Div(
            className="sidebar-spacer"
        ),

        html.Div(
            [
                html.Div(
                    "DATASET",
                    className="side-label"
                ),

                html.Div(
                    "Trạm quan trắc · Hạ Long",
                    className="side-foot"
                ),

                html.Div(
                    "UTC+7 · OBS mm/1h",
                    className="side-foot muted"
                )
            ],
            className="sidebar-meta"
        )
    ],
    className="sidebar"
)


# ============================================================
# 6. HEADER
# ============================================================

header = html.Header(
    [
        html.Div(
            [
                html.Div(
                    "RAINFALL MODEL PERFORMANCE",
                    className="eyebrow"
                ),

                html.H1(
                    "WeatherPlus Nowcast",
                    className="page-title"
                ),

                html.P(
                    "Đánh giá chất lượng dự báo lượng mưa "
                    "tại trạm Hạ Long, Quảng Ninh",
                    className="page-desc"
                )
            ]
        ),

        html.Div(
            [
                html.Div(
                    "●  DATA READY",
                    className="status-pill"
                ),

                html.Div(
                    id="data-range-label",
                    className="date-range-label"
                )
            ],
            className="header-meta"
        )
    ],
    className="topbar"
)


# ============================================================
# 7. FILTER BAR
# ============================================================

filter_bar = html.Section(
    [
        html.Div(
            [
                html.Label(
                    "Hạn dự báo",
                    className="filter-label"
                ),

                dcc.Dropdown(
                    HORIZONS,
                    "1h",
                    id="horizon-filter",
                    clearable=False,
                    className="filter-control"
                )
            ]
        ),

        html.Div(
            [
                html.Label(
                    "Khoảng thời gian (giờ Việt Nam)",
                    className="filter-label"
                ),

                dcc.DatePickerRange(
                    id="date-filter",
                    min_date_allowed=MIN_DATE,
                    max_date_allowed=MAX_DATE,
                    start_date=MIN_DATE,
                    end_date=MAX_DATE,
                    display_format="DD/MM/YYYY",
                    className="date-control"
                )
            ]
        ),

        html.Div(
            [
                html.Label(
                    "Mùa",
                    className="filter-label"
                ),

                dcc.Dropdown(
                    [
                        {"label": s, "value": s}
                        for s in SEASON_ORDER
                    ],
                    SEASON_ORDER,
                    id="season-filter",
                    multi=True,
                    className="filter-control"
                )
            ]
        ),

        html.Div(
            [
                html.Label(
                    "Nhóm cường độ OBS",
                    className="filter-label"
                ),

                dcc.Dropdown(
                    [
                        {"label": c, "value": c}
                        for c in CLASS_ORDER
                    ],
                    CLASS_ORDER,
                    id="class-filter",
                    multi=True,
                    className="filter-control"
                )
            ]
        )
    ],
    className="filter-bar"
)


# ============================================================
# 8. KPI CARDS
# ============================================================

kpi_grid = html.Div(
    [
        card(
            "SỐ MẪU ĐÁNH GIÁ",
            "kpi-n",
            "Cặp Nowcast – OBS hợp lệ",
            "blue"
        ),

        card(
            "MAE",
            "kpi-mae",
            "Sai số tuyệt đối trung bình · mm",
            "cyan"
        ),

        card(
            "RMSE",
            "kpi-rmse",
            "Căn bậc hai sai số bình phương · mm",
            "violet"
        ),

        card(
            "BIAS",
            "kpi-bias",
            "Nowcast − OBS · mm",
            "amber"
        ),

        card(
            "POD",
            "kpi-pod",
            "Khả năng phát hiện mưa ≥ 0,2 mm",
            "green"
        ),

        card(
            "FAR",
            "kpi-far",
            "Tỷ lệ cảnh báo sai trong dự báo mưa",
            "rose"
        ),

        card(
            "CSI",
            "kpi-csi",
            "Critical Success Index · ngưỡng 0,2 mm",
            "dark"
        )
    ],
    className="kpi-grid"
)


# ============================================================
# 9. TIME ANALYSIS
# ============================================================

time_analysis = html.Div(
    [
        panel(
            "Diễn biến lượng mưa theo thời gian",
            "Chuỗi OBS và Nowcast tại cùng valid_time · đơn vị mm/1h",
            "timeseries",
            "panel panel-wide",
            graph_height=350
        ),

        panel(
            "Phân bố sai số",
            "Sai số = Nowcast − OBS; đường 0 biểu thị dự báo không lệch",
            "error-dist",
            graph_height=350
        )
    ],
    id="time-analysis",
    className="grid-main"
)


# ============================================================
# 10. INTENSITY ANALYSIS
# ============================================================

intensity_analysis = html.Div(
    [
        panel(
            "Confusion matrix · 5 cấp độ",
            "Hàng: lớp OBS · Cột: lớp Nowcast · số lượng mẫu",
            "confusion",
            graph_height=390
        ),

        panel(
            "Metrics theo cường độ OBS",
            "POD, FAR và CSI được tính one-vs-rest cho từng lớp",
            "class-metrics",
            graph_height=390
        )
    ],
    id="intensity-analysis",
    className="grid-main"
)


# ============================================================
# 11. SEASONAL AND HOURLY ANALYSIS
# ============================================================

season_hour_analysis = html.Div(
    [
        panel(
            "Sai số theo mùa",
            "MAE và RMSE theo mùa khí hậu; "
            "mùa được xác định từ valid_time UTC",
            "season-chart",
            graph_height=320
        ),

        panel(
            "Hiệu suất theo giờ địa phương",
            "Giờ Việt Nam (UTC+7) · theo thời điểm valid_time",
            "hour-chart",
            graph_height=320
        )
    ],
    className="grid-main"
)


# ============================================================
# 12. METRICS TABLE
# ============================================================

metrics_table = html.Section(
    [
        html.Div(
            [
                html.H3(
                    "Bảng đánh giá theo cường độ"
                ),

                html.P(
                    "N là số mẫu OBS thuộc nhóm; "
                    "nhóm N < 30 cần được diễn giải thận trọng.",
                    className="panel-subtitle"
                )
            ]
        ),

        dash_table.DataTable(
            id="metrics-table",

            columns=[
                {
                    "name": "Cường độ OBS",
                    "id": "Intensity"
                },

                {
                    "name": "N",
                    "id": "N",
                    "type": "numeric"
                },

                {
                    "name": "MAE (mm)",
                    "id": "MAE",
                    "type": "numeric",
                    "format": {"specifier": ".3f"}
                },

                {
                    "name": "RMSE (mm)",
                    "id": "RMSE",
                    "type": "numeric",
                    "format": {"specifier": ".3f"}
                },

                {
                    "name": "Bias (mm)",
                    "id": "Bias",
                    "type": "numeric",
                    "format": {"specifier": ".3f"}
                },

                {
                    "name": "POD",
                    "id": "POD",
                    "type": "numeric",
                    "format": {"specifier": ".3f"}
                },

                {
                    "name": "FAR",
                    "id": "FAR",
                    "type": "numeric",
                    "format": {"specifier": ".3f"}
                },

                {
                    "name": "CSI",
                    "id": "CSI",
                    "type": "numeric",
                    "format": {"specifier": ".3f"}
                },

                {
                    "name": "Cảnh báo",
                    "id": "Warning"
                }
            ],

            data=[],
            page_size=10,
            sort_action="native",

            style_table={
                "overflowX": "auto"
            },

            style_header={
                "backgroundColor": "#F1F5F9",
                "fontWeight": "700",
                "color": "#334155",
                "border": "none"
            },

            style_cell={
                "padding": "12px 14px",
                "fontFamily": "inherit",
                "fontSize": "13px",
                "border": "none",
                "borderBottom": "1px solid #E8EDF4",
                "textAlign": "left"
            },

            style_data_conditional=[
                {
                    "if": {
                        "filter_query": '{Warning} = "N < 30"',
                        "column_id": "Warning"
                    },
                    "color": "#B45309",
                    "fontWeight": "700"
                }
            ]
        )
    ],
    className="panel table-panel"
)


# ============================================================
# 13. MAIN LAYOUT
# ============================================================

app.layout = html.Div(
    [
        sidebar,

        html.Main(
            [
                header,

                filter_bar,

                kpi_grid,

                time_analysis,

                intensity_analysis,

                season_hour_analysis,

                metrics_table,

                html.Footer(
                    "WeatherPlus Nowcast · Rainfall Verification "
                    "· Hạ Long, Quảng Ninh",
                    className="footer"
                )
            ],
            id="overview",
            className="main-content"
        )
    ],
    className="app-shell"
)


# ============================================================
# 14. FIGURE HELPERS
# ============================================================

def empty_fig(
    message="Không có dữ liệu với bộ lọc hiện tại"
):

    fig = go.Figure()

    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={
            "color": "#64748B",
            "size": 14
        }
    )

    fig.update_layout(
        template="plotly_white",
        height=320,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        )
    )

    return fig


def style_fig(fig, height=330):

    fig.update_layout(
        template="plotly_white",

        height=height,

        margin=dict(
            l=24,
            r=20,
            t=22,
            b=35
        ),

        font=dict(
            family="Be Vietnam Pro, sans-serif",
            color="#475569",
            size=11
        ),

        paper_bgcolor="white",
        plot_bgcolor="white",

        legend=dict(
            orientation="h",
            y=1.12,
            x=0
        ),

        hovermode="x unified"
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor="#E8EDF4"
    )

    fig.update_yaxes(
        gridcolor="#EEF2F7",
        zerolinecolor="#CBD5E1"
    )

    return fig


# ============================================================
# 15. DASHBOARD CALLBACK
# ============================================================

@app.callback(
    Output("data-range-label", "children"),

    Output("kpi-n", "children"),
    Output("kpi-mae", "children"),
    Output("kpi-rmse", "children"),
    Output("kpi-bias", "children"),
    Output("kpi-pod", "children"),
    Output("kpi-far", "children"),
    Output("kpi-csi", "children"),

    Output("timeseries", "figure"),
    Output("error-dist", "figure"),
    Output("confusion", "figure"),
    Output("class-metrics", "figure"),
    Output("season-chart", "figure"),
    Output("hour-chart", "figure"),

    Output("metrics-table", "data"),

    Input("horizon-filter", "value"),
    Input("date-filter", "start_date"),
    Input("date-filter", "end_date"),
    Input("season-filter", "value"),
    Input("class-filter", "value")
)


def update_dashboard(
    horizon,
    start_date,
    end_date,
    seasons,
    classes
):

    # --------------------------------------------------------
    # A. BASE FILTERS
    # --------------------------------------------------------

    d_base = df[
        df["Horizon"] == horizon
    ].copy()

    if start_date:

        d_base = d_base[
            d_base["valid_time_local"].dt.date
            >= pd.to_datetime(start_date).date()
        ]

    if end_date:

        d_base = d_base[
            d_base["valid_time_local"].dt.date
            <= pd.to_datetime(end_date).date()
        ]

    if seasons:

        d_base = d_base[
            d_base["season"].isin(seasons)
        ]

    # --------------------------------------------------------
    # B. INTENSITY FILTER
    # --------------------------------------------------------

    if classes:

        d = d_base[
            d_base["obs_class"].isin(classes)
        ].copy()

    else:

        d = d_base.copy()

    # --------------------------------------------------------
    # C. EMPTY DATA
    # --------------------------------------------------------

    if d.empty:

        e = empty_fig()

        return (
            "Không có dữ liệu",

            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",

            e,
            e,
            e,
            e,
            e,
            e,

            []
        )

    # --------------------------------------------------------
    # D. KPI CALCULATION
    # --------------------------------------------------------

    obs = d["rain_eval"].to_numpy(float)
    fc = d["nowcast"].to_numpy(float)

    err = fc - obs

    n = len(d)

    mae = float(
        np.mean(np.abs(err))
    )

    rmse = float(
        np.sqrt(np.mean(err ** 2))
    )

    bias = float(
        np.mean(err)
    )

    # Binary event threshold
    obs_event = obs >= 0.2
    fc_event = fc >= 0.2

    hit = int(
        np.sum(obs_event & fc_event)
    )

    miss = int(
        np.sum(obs_event & ~fc_event)
    )

    fa = int(
        np.sum(~obs_event & fc_event)
    )

    pod = (
        hit / (hit + miss)
        if hit + miss
        else np.nan
    )

    far = (
        fa / (hit + fa)
        if hit + fa
        else np.nan
    )

    csi = (
        hit / (hit + miss + fa)
        if hit + miss + fa
        else np.nan
    )

    kpi = [
        f"{n:,}",

        f"{mae:.3f}",

        f"{rmse:.3f}",

        f"{bias:+.3f}",

        f"{pod:.3f}"
        if np.isfinite(pod)
        else "—",

        f"{far:.3f}"
        if np.isfinite(far)
        else "—",

        f"{csi:.3f}"
        if np.isfinite(csi)
        else "—"
    ]

    date_label = (
        f"{d['valid_time_local'].min():%d/%m/%Y} — "
        f"{d['valid_time_local'].max():%d/%m/%Y} · {horizon}"
    )

    # --------------------------------------------------------
    # E. TIME SERIES
    # --------------------------------------------------------

    ts = d.sort_values(
        "valid_time_local"
    )

    fig_ts = go.Figure()

    fig_ts.add_trace(
        go.Scatter(
            x=ts["valid_time_local"],
            y=ts["rain_eval"],
            name="OBS",
            mode="lines",
            line=dict(
                color="#0F766E",
                width=1.7
            )
        )
    )

    fig_ts.add_trace(
        go.Scatter(
            x=ts["valid_time_local"],
            y=ts["nowcast"],
            name="Nowcast",
            mode="lines",
            line=dict(
                color="#2563EB",
                width=1.4
            )
        )
    )

    fig_ts.update_layout(
        yaxis_title="Lượng mưa (mm/1h)",
        xaxis_title="Valid time (giờ Việt Nam)"
    )

    style_fig(
        fig_ts,
        350
    )

    # --------------------------------------------------------
    # F. ERROR DISTRIBUTION
    # --------------------------------------------------------

    fig_err = px.histogram(
        d,
        x="error",
        nbins=45,
        color_discrete_sequence=["#4F46E5"]
    )

    fig_err.add_vline(
        x=0,
        line_dash="dash",
        line_color="#E11D48"
    )

    fig_err.update_layout(
        xaxis_title="Sai số (mm)",
        yaxis_title="Số mẫu",
        showlegend=False
    )

    style_fig(
        fig_err,
        350
    )

    # --------------------------------------------------------
    # G. CONFUSION MATRIX
    # --------------------------------------------------------
    # Use d_base rather than intensity-filtered d.
    # This keeps all OBS intensity classes visible.

    cm = pd.crosstab(
        pd.Categorical(
            d_base["obs_class"],
            categories=CLASS_ORDER,
            ordered=True
        ),

        pd.Categorical(
            d_base["forecast_class"],
            categories=CLASS_ORDER,
            ordered=True
        ),

        dropna=False
    ).reindex(
        index=CLASS_ORDER,
        columns=CLASS_ORDER,
        fill_value=0
    )

    z = cm.to_numpy()

    fig_cm = go.Figure(
        go.Heatmap(
            z=z,
            x=CLASS_ORDER,
            y=CLASS_ORDER,
            colorscale="Blues",
            text=z,
            texttemplate="%{text}",

            hovertemplate=(
                "OBS: %{y}<br>"
                "Nowcast: %{x}<br>"
                "N: %{z}<extra></extra>"
            ),

            colorbar={
                "title": "N"
            }
        )
    )

    fig_cm.update_layout(
        xaxis_title="Lớp Nowcast",
        yaxis_title="Lớp OBS"
    )

    style_fig(
        fig_cm,
        390
    )

    # --------------------------------------------------------
    # H. CLASSIFICATION METRICS
    # --------------------------------------------------------
    # One-vs-rest metrics calculated on d_base.

    class_rows = []

    for c in CLASS_ORDER:

        g = d_base[
            d_base["obs_class"] == c
        ]

        tp = int(
            (
                (d_base["obs_class"] == c)
                &
                (d_base["forecast_class"] == c)
            ).sum()
        )

        fn = int(
            (d_base["obs_class"] == c).sum() - tp
        )

        fp = int(
            (d_base["forecast_class"] == c).sum() - tp
        )

        pod_c = (
            tp / (tp + fn)
            if tp + fn
            else np.nan
        )

        far_c = (
            fp / (tp + fp)
            if tp + fp
            else np.nan
        )

        csi_c = (
            tp / (tp + fn + fp)
            if tp + fn + fp
            else np.nan
        )

        class_rows.append(
            {
                "Intensity": c,

                "N": len(g),

                "MAE": (
                    np.mean(
                        np.abs(
                            g["nowcast"] - g["rain_eval"]
                        )
                    )
                    if len(g)
                    else np.nan
                ),

                "RMSE": (
                    np.sqrt(
                        np.mean(
                            (
                                g["nowcast"] - g["rain_eval"]
                            ) ** 2
                        )
                    )
                    if len(g)
                    else np.nan
                ),

                "Bias": (
                    np.mean(
                        g["nowcast"] - g["rain_eval"]
                    )
                    if len(g)
                    else np.nan
                ),

                "POD": pod_c,
                "FAR": far_c,
                "CSI": csi_c,

                "Warning": (
                    "N < 30"
                    if len(g) < 30
                    else "—"
                )
            }
        )

    class_df = pd.DataFrame(
        class_rows
    )

    # --------------------------------------------------------
    # I. CLASS METRICS CHART
    # --------------------------------------------------------

    fig_class = go.Figure()

    for metric, color in [
        ("POD", "#0F766E"),
        ("FAR", "#EA580C"),
        ("CSI", "#4F46E5")
    ]:

        fig_class.add_trace(
            go.Bar(
                x=class_df["Intensity"],
                y=class_df[metric],
                name=metric,
                marker_color=color
            )
        )

    fig_class.update_layout(
        barmode="group",

        yaxis=dict(
            title="Score",
            range=[0, 1]
        )
    )

    style_fig(
        fig_class,
        390
    )

    # --------------------------------------------------------
    # J. SEASONAL METRICS
    # --------------------------------------------------------

    season_rows = []

    for season in SEASON_ORDER:

        g = d[
            d["season"] == season
        ]

        if len(g):

            e = (
                g["nowcast"] - g["rain_eval"]
            )

            season_rows.append(
                {
                    "Season": season,

                    "MAE": np.mean(
                        np.abs(e)
                    ),

                    "RMSE": np.sqrt(
                        np.mean(e ** 2)
                    ),

                    "N": len(g)
                }
            )

    season_df = pd.DataFrame(
        season_rows
    )

    fig_season = go.Figure()

    if not season_df.empty:

        fig_season.add_trace(
            go.Bar(
                x=season_df["Season"],
                y=season_df["MAE"],
                name="MAE",
                marker_color="#0EA5E9"
            )
        )

        fig_season.add_trace(
            go.Bar(
                x=season_df["Season"],
                y=season_df["RMSE"],
                name="RMSE",
                marker_color="#6366F1"
            )
        )

        fig_season.update_layout(
            barmode="group",
            yaxis_title="Sai số (mm)"
        )

        style_fig(
            fig_season,
            320
        )

    else:

        fig_season = empty_fig()

    # --------------------------------------------------------
    # K. HOURLY CSI
    # --------------------------------------------------------

    hour_rows = []

    for hour in range(24):

        g = d[
            d["hour_local"] == hour
        ]

        if len(g):

            oe = (
                g["rain_eval"].to_numpy() >= 0.2
            )

            fe = (
                g["nowcast"].to_numpy() >= 0.2
            )

            h = int(
                np.sum(oe & fe)
            )

            m = int(
                np.sum(oe & ~fe)
            )

            f = int(
                np.sum(~oe & fe)
            )

            hour_rows.append(
                {
                    "Hour": hour,

                    "CSI": (
                        h / (h + m + f)
                        if h + m + f
                        else np.nan
                    ),

                    "N": len(g)
                }
            )

    hour_df = pd.DataFrame(
        hour_rows
    )

    fig_hour = go.Figure()

    if not hour_df.empty:

        fig_hour.add_trace(
            go.Scatter(
                x=hour_df["Hour"],
                y=hour_df["CSI"],
                mode="lines+markers",
                name="CSI",

                line=dict(
                    color="#0F766E",
                    width=2.5
                )
            )
        )

        fig_hour.update_layout(
            xaxis=dict(
                title="Giờ Việt Nam",
                tickmode="linear",
                dtick=2
            ),

            yaxis=dict(
                title="CSI",
                range=[0, 1]
            )
        )

        style_fig(
            fig_hour,
            320
        )

    else:

        fig_hour = empty_fig()

    # --------------------------------------------------------
    # L. RETURN OUTPUTS
    # --------------------------------------------------------

    return (
        date_label,

        *kpi,

        fig_ts,
        fig_err,
        fig_cm,
        fig_class,
        fig_season,
        fig_hour,

        class_df.replace(
            {np.nan: None}
        ).to_dict("records")
    )


# ============================================================
# 16. RUN SERVER
# ============================================================

# WSGI entry point used by Render and other production hosts.
server = app.server


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    app.run(
        debug=False,
        host="0.0.0.0",
        port=port
    )
