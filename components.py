import plotly.express as px
import pandas as pd

def timeseries_chart(df, date_col, metric_col, variant_col):
    if date_col not in df.columns:
        return None
    daily = df.groupby([date_col, variant_col], as_index=False)[metric_col].mean()
    fig = px.line(daily, x=date_col, y=metric_col, color=variant_col, markers=True)
    fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=320)
    return fig

def dist_chart(df, metric_col, variant_col):
    fig = px.histogram(df, x=metric_col, color=variant_col, barmode="overlay", opacity=0.6, nbins=30)
    fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=300)
    return fig