"""
AWS vs GCP — Executive KPI Report (trailing 24 months)
======================================================

Combines the AWS Cost Explorer extracts (FinOps_AWS/Monthly_Cost_For_*.csv) with the
GCP BigQuery billing export (cached to FinOps_GCP/GCP_Billing_History.csv) and builds
a two-year KPI scorecard: Year 1 = months 24..13 back, Year 2 = last 12 months.

Both clouds are split the same way:
  * Cloud Services — first-party AWS / Google services (what the budgets track)
  * Marketplace    — third-party software billed through the cloud (AWS Marketplace,
                     GCP THIRD_PARTY_RESELLER / THIRD_PARTY_AGENCY)
  * Tax
GCP cost is NET: cost + credits (CUD, subscription benefits, promotions, discounts).

Usage
-----
    python multicloud_kpis.py                    # uses cached GCP history
    python multicloud_kpis.py --refresh-gcp      # re-query BigQuery first (~15 GB scanned)
"""
from __future__ import annotations

import argparse
import html
import os
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta

import finops_exec_report as fr
from finops_exec_report import _money, _pct, _layout, _table, _fig_html

HERE = os.path.dirname(os.path.abspath(__file__))
AWS_DIR = HERE
GCP_DIR = os.path.join(os.path.dirname(HERE), 'FinOps_GCP')
GCP_HISTORY = 'GCP_Billing_History.csv'
GCP_CREDITS = 'GCP_Credits_History.csv'

GCP_PROJECT = 'billing-query-371121'
GCP_TABLE = 'billing-query-371121.all_billing_data.gcp_billing_export_v1_01C5AB_0209B1_81C892'

AWS_MONTHLY_BUDGET = 550_000

SVC, MKT, TAX = 'Cloud Services', 'Marketplace', 'Tax'
AWS, GCP = 'AWS', 'GCP'

# Palette: validated categorical slots in fixed order (GCP=slot1 blue, AWS=slot2 orange,
# GCP Marketplace=slot3 aqua, AWS Marketplace=slot4 yellow), Tax=neutral gray. Theme-aware.
def set_theme(name: str = 'dark') -> None:
    """Switch both report modules between 'dark' and 'light'."""
    global COLORS, C_TAX, C_PROVIDER, INK, INK2, MUTED, GRID, AXIS, BAD, GOOD, CSS
    fr.set_theme(name)
    s = fr.SERIES
    COLORS = {(GCP, SVC): s[0], (AWS, SVC): s[1], (GCP, MKT): s[2], (AWS, MKT): s[3]}
    C_PROVIDER = {GCP: s[0], AWS: s[1]}
    C_TAX = fr.C_OTHER
    INK, INK2, MUTED, GRID, AXIS, BAD, GOOD = fr.INK, fr.INK2, fr.MUTED, fr.GRID, fr.AXIS, fr.BAD, fr.GOOD
    CSS = fr.css()


set_theme(fr.THEME)

GENAI_PATTERN = {AWS: r'Bedrock|Claude|SageMaker|Amazon Q|Kendra|Personalize',
                 GCP: r'Gemini|Vertex AI|Generative|Dialogflow|Document AI'}


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def refresh_gcp_history(end_month: str | None = None, months: int = 24, gcp_dir: str = GCP_DIR) -> str:
    """Re-query the BigQuery billing export and overwrite the cached CSVs.

    Needs `gcloud auth application-default login`. Scans ~15 GB (a few cents on-demand).
    """
    from google.cloud import bigquery

    end = pd.Timestamp(end_month) if end_month else (pd.Timestamp.today().normalize().replace(day=1) - relativedelta(months=1))
    start = end - relativedelta(months=months - 1)
    m0, m1 = start.strftime('%Y%m'), end.strftime('%Y%m')
    client = bigquery.Client(project=GCP_PROJECT)
    main = f"""
    SELECT invoice.month AS invoice_month, service.description AS service,
           IFNULL(project.name, '(no project)') AS project,
           IFNULL(transaction_type, 'GOOGLE') AS transaction_type, IFNULL(cost_type, 'regular') AS cost_type,
           EXISTS(SELECT 1 FROM UNNEST(labels) l WHERE LOWER(l.key) IN ('env', 'environment')) AS has_env_label,
           SUM(cost) AS cost,
           SUM(IFNULL((SELECT SUM(x.amount) FROM UNNEST(credits) x), 0)) AS credits,
           SUM(IFNULL(cost_at_list, cost)) AS list_cost
    FROM `{GCP_TABLE}`
    WHERE invoice.month BETWEEN '{m0}' AND '{m1}'
    GROUP BY 1, 2, 3, 4, 5, 6"""
    credits = f"""
    SELECT invoice.month AS invoice_month, x.type AS credit_type, SUM(x.amount) AS amount
    FROM `{GCP_TABLE}`, UNNEST(credits) x
    WHERE invoice.month BETWEEN '{m0}' AND '{m1}'
    GROUP BY 1, 2"""
    client.query(main).to_dataframe().to_csv(os.path.join(gcp_dir, GCP_HISTORY), index=False)
    client.query(credits).to_dataframe().to_csv(os.path.join(gcp_dir, GCP_CREDITS), index=False)
    return os.path.join(gcp_dir, GCP_HISTORY)


def load_gcp(gcp_dir: str = GCP_DIR) -> pd.DataFrame:
    g = pd.read_csv(os.path.join(gcp_dir, GCP_HISTORY), dtype={'invoice_month': str})
    g['Date'] = pd.to_datetime(g['invoice_month'], format='%Y%m')
    g['Cost'] = g['cost'] + g['credits']                                   # net of credits
    g['Category'] = np.select([g['cost_type'].eq('tax'), g['transaction_type'].ne('GOOGLE')], [TAX, MKT], SVC)
    g['Provider'] = GCP
    g['Service'] = g['service']
    g['Taggable'] = g['Category'].eq(SVC)
    g['Tagged'] = g['has_env_label'].astype(bool)
    return g[['Provider', 'Date', 'Category', 'Service', 'project', 'Cost', 'list_cost', 'Taggable', 'Tagged']]


def load_aws(aws_dir: str = AWS_DIR) -> pd.DataFrame:
    a = fr.load_cost_data(aws_dir)
    a['Category'] = a['Category'].map({fr.AWS: SVC, fr.MKT: MKT, fr.TAX: TAX})
    a['Provider'] = AWS
    a['Taggable'] = a['Category'].eq(SVC) & ~a['Service'].isin(fr.NON_TAGGABLE) & a['Env'].notna()
    a['Tagged'] = a['Env'].notna() & a['Env'].ne('untagged')
    a['project'] = None
    a['list_cost'] = np.nan
    return a[['Provider', 'Date', 'Category', 'Service', 'project', 'Cost', 'list_cost', 'Taggable', 'Tagged']]


def load_all(aws_dir: str = AWS_DIR, gcp_dir: str = GCP_DIR) -> pd.DataFrame:
    return pd.concat([load_aws(aws_dir), load_gcp(gcp_dir)], ignore_index=True)


# --------------------------------------------------------------------------------------
# KPIs
# --------------------------------------------------------------------------------------
@dataclass
class KPIData:
    df: pd.DataFrame
    end: pd.Timestamp
    y1: tuple
    y2: tuple
    budget: float
    k: dict = field(default_factory=dict)


def _window(df, w):
    return df[(df['Date'] >= w[0]) & (df['Date'] <= w[1])]


def _yr_label(w):
    return f'{w[0]:%b %y}–{w[1]:%b %y}'


def compute(df: pd.DataFrame, end_month: str | None = None, budget: float = AWS_MONTHLY_BUDGET) -> KPIData:
    # latest month BOTH clouds have
    common_end = min(df[df['Provider'] == p]['Date'].max() for p in (AWS, GCP))
    end = pd.Timestamp(end_month) if end_month else common_end
    y2 = (end - relativedelta(months=11), end)
    y1 = (end - relativedelta(months=23), end - relativedelta(months=12))
    df = df[(df['Date'] >= y1[0]) & (df['Date'] <= end)].copy()
    kd = KPIData(df=df, end=end, y1=y1, y2=y2, budget=budget)
    k = kd.k

    monthly = df.pivot_table(index='Date', columns=['Provider', 'Category'], values='Cost', aggfunc='sum').fillna(0)
    k['monthly'] = monthly
    svc = monthly.xs(SVC, axis=1, level='Category')
    k['svc'] = svc

    rows = []
    def add(name, fmt, aws1, aws2, gcp1, gcp2, c1=None, c2=None, higher_is_bad=True, note=''):
        rows.append(dict(KPI=name, fmt=fmt, AWS_Y1=aws1, AWS_Y2=aws2, GCP_Y1=gcp1, GCP_Y2=gcp2,
                         ALL_Y1=c1, ALL_Y2=c2, bad_up=higher_is_bad, note=note))

    def tot(p, w, cat=None):
        d = _window(df, w)
        d = d[d['Provider'] == p] if p else d
        d = d[d['Category'] == cat] if cat else d
        return d['Cost'].sum()

    for label, cat in [('Total spend (all-in)', None), ('Cloud Services', SVC), ('Marketplace', MKT), ('Tax', TAX)]:
        add(label, 'money', tot(AWS, y1, cat), tot(AWS, y2, cat), tot(GCP, y1, cat), tot(GCP, y2, cat),
            tot(None, y1, cat), tot(None, y2, cat))

    # run-rate: average of the last 3 months of each year window (Cloud Services)
    def rr(p, w):
        s = svc[p] if p else svc.sum(axis=1)
        return s.loc[w[1] - relativedelta(months=2):w[1]].mean()
    add('Cloud Services run-rate (3-mo avg / month)', 'money', rr(AWS, y1), rr(AWS, y2), rr(GCP, y1), rr(GCP, y2),
        rr(None, y1), rr(None, y2))

    # provider share of Cloud Services
    s1, s2 = tot(None, y1, SVC), tot(None, y2, SVC)
    add('Share of Cloud Services spend', 'pct', tot(AWS, y1, SVC) / s1, tot(AWS, y2, SVC) / s2,
        tot(GCP, y1, SVC) / s1, tot(GCP, y2, SVC) / s2, higher_is_bad=None)

    # Marketplace share of all-in
    def mkt_share(p, w):
        return tot(p, w, MKT) / tot(p, w)
    add('Marketplace share of total', 'pct', mkt_share(AWS, y1), mkt_share(AWS, y2), mkt_share(GCP, y1), mkt_share(GCP, y2),
        mkt_share(None, y1), mkt_share(None, y2))

    # Volatility of Cloud Services month to month (coefficient of variation)
    def cv(p, w):
        s = (svc[p] if p else svc.sum(axis=1)).loc[w[0]:w[1]]
        return s.std() / s.mean()
    add('Monthly volatility (CV of Cloud Services)', 'pct', cv(AWS, y1), cv(AWS, y2), cv(GCP, y1), cv(GCP, y2),
        cv(None, y1), cv(None, y2), note='std ÷ mean; lower = more predictable')

    # Top-5 service concentration
    def conc(p, w):
        d = _window(df, w)
        d = d[(d['Provider'] == p) & (d['Category'] == SVC)].groupby('Service')['Cost'].sum()
        return d.nlargest(5).sum() / d.sum()
    add('Top-5 services concentration', 'pct', conc(AWS, y1), conc(AWS, y2), conc(GCP, y1), conc(GCP, y2), higher_is_bad=None)

    # Support as % of Cloud Services
    def support(p, w):
        d = _window(df, w)
        d = d[(d['Provider'] == p) & (d['Category'] == SVC)]
        return d[d['Service'].str.contains('Support')]['Cost'].sum() / d['Cost'].sum()
    add('Support as % of Cloud Services', 'pct', support(AWS, y1), support(AWS, y2), support(GCP, y1), support(GCP, y2))

    # GenAI spend (all categories — Claude on Bedrock is Marketplace-billed)
    def genai(p, w):
        d = _window(df, w)
        d = d[d['Provider'] == p]
        return d[d['Service'].str.contains(GENAI_PATTERN[p], regex=True)]['Cost'].sum()
    add('GenAI / ML spend', 'money', genai(AWS, y1), genai(AWS, y2), genai(GCP, y1), genai(GCP, y2),
        genai(AWS, y1) + genai(GCP, y1), genai(AWS, y2) + genai(GCP, y2), higher_is_bad=None,
        note='AWS: Bedrock/Claude, SageMaker, Q… · GCP: Gemini, Vertex AI…')

    # Environment tag / label coverage (taggable Cloud Services, months where captured)
    def cov(p, w):
        d = _window(df, w)
        d = d[(d['Provider'] == p) & d['Taggable']]
        return (d['Cost'] * d['Tagged']).sum() / d['Cost'].sum() if len(d) and d['Cost'].sum() else np.nan
    add('Env tag / label coverage', 'pct', cov(AWS, y1), cov(AWS, y2), cov(GCP, y1), cov(GCP, y2), higher_is_bad=False,
        note='share of taggable spend with an env tag (AWS) / env label (GCP)')

    # GCP effective discount vs list
    def disc(w):
        d = _window(df, w)
        d = d[(d['Provider'] == GCP) & (d['Category'] == SVC)]
        return 1 - d['Cost'].sum() / d['list_cost'].sum()
    add('Effective discount vs list price', 'pct', np.nan, np.nan, disc(y1), disc(y2), higher_is_bad=False,
        note='GCP only (AWS extract has no list price)')

    # AWS budget
    aws_y2 = svc[AWS].loc[y2[0]:y2[1]]
    add(f'AWS months within {_money(budget)} budget', 'count', int((svc[AWS].loc[y1[0]:y1[1]] <= budget).sum()),
        int((aws_y2 <= budget).sum()), np.nan, np.nan, higher_is_bad=False)

    k['scorecard'] = pd.DataFrame(rows)

    # Service movers (Y2 vs Y1, $/month avg) per provider
    def movers(p):
        d = df[(df['Provider'] == p) & (df['Category'] == SVC)]
        a = _window(d, y1).groupby('Service')['Cost'].sum() / 12
        b = _window(d, y2).groupby('Service')['Cost'].sum() / 12
        m = pd.DataFrame({'Y1 avg/mo': a, 'Y2 avg/mo': b}).fillna(0)
        m['Δ/mo'] = m['Y2 avg/mo'] - m['Y1 avg/mo']
        peak = _window(d, y2).groupby(['Service', 'Date'])['Cost'].sum().groupby('Service').max()
        m['one_time'] = (peak / (m['Y2 avg/mo'] * 12)).reindex(m.index).fillna(0) > 0.7
        return m.sort_values('Δ/mo')
    k['movers'] = {AWS: movers(AWS), GCP: movers(GCP)}

    # Largest single-month charges (lumpy / one-time) — any category
    sm = df.groupby(['Provider', 'Category', 'Service', 'Date'])['Cost'].sum().reset_index()
    med = sm.groupby(['Provider', 'Service'])['Cost'].transform('median')
    lumpy = sm[(sm['Cost'] >= 100_000) & (sm['Cost'] > 3 * med.clip(lower=1))]
    k['lumpy'] = lumpy.sort_values('Cost', ascending=False)

    # GenAI monthly
    ga = {}
    for p in (AWS, GCP):
        d = df[(df['Provider'] == p) & df['Service'].str.contains(GENAI_PATTERN[p], regex=True)]
        ga[p] = d.groupby('Date')['Cost'].sum().reindex(svc.index, fill_value=0)
    k['genai'] = pd.DataFrame(ga)

    # Tag coverage monthly
    tc = {}
    for p in (AWS, GCP):
        d = df[(df['Provider'] == p) & df['Taggable']]
        g = d.assign(t=d['Cost'] * d['Tagged']).groupby('Date')[['t', 'Cost']].sum()
        tc[p] = g['t'] / g['Cost']
    k['tagcov'] = pd.DataFrame(tc)

    _insights(kd)
    return kd


def _insights(kd: KPIData) -> None:
    k, sc = kd.k, kd.k['scorecard'].set_index('KPI')
    out = []
    t = sc.loc['Total spend (all-in)']
    out.append(f'Combined cloud spend was {_money(t.ALL_Y2)} in the last 12 months vs {_money(t.ALL_Y1)} the year before '
               f'({_pct(t.ALL_Y2 / t.ALL_Y1 - 1)}). AWS {_pct(t.AWS_Y2 / t.AWS_Y1 - 1)}, GCP {_pct(t.GCP_Y2 / t.GCP_Y1 - 1)}.')
    rr = sc.loc['Cloud Services run-rate (3-mo avg / month)']
    out.append(f'Cloud Services run-rate: AWS {_money(rr.AWS_Y1)} → {_money(rr.AWS_Y2)}/mo, GCP {_money(rr.GCP_Y1)} → '
               f'{_money(rr.GCP_Y2)}/mo. GCP is now {rr.GCP_Y2 / rr.AWS_Y2:.0%} the size of AWS (was {rr.GCP_Y1 / rr.AWS_Y1:.0%}).')
    for p in (GCP, AWS):
        top = k['movers'][p].tail(3)
        out.append(f'{p} biggest increases (avg $/mo, Y2 vs Y1): ' +
                   ', '.join(f'{fr._short(s, 28)} +{_money(v)}' + (' (one-time charge)' if ot else '')
                             for s, v, ot in zip(top.index[::-1], top['Δ/mo'][::-1], top['one_time'][::-1])) + '.')
        low = k['movers'][p].head(2)
        if (low['Δ/mo'] < 0).any():
            out.append(f'{p} biggest decreases: ' +
                       ', '.join(f'{fr._short(s, 28)} {_money(v)}' for s, v in low['Δ/mo'].items() if v < 0) + '.')
    ga = sc.loc['GenAI / ML spend']
    out.append(f'GenAI/ML spend: {_money(ga.ALL_Y1)} → {_money(ga.ALL_Y2)} '
               f'(GCP {_money(ga.GCP_Y2)}, AWS {_money(ga.AWS_Y2)} in the last 12 months).')
    cv = sc.loc['Env tag / label coverage']
    out.append(f'Only {cv.AWS_Y2:.0%} of taggable AWS spend and {cv.GCP_Y2:.0%} of GCP spend carry an environment '
               'tag/label — cost allocation by environment is not possible today for most spend.')
    m = sc.loc['Marketplace']
    out.append(f'Marketplace (third-party) was {_money(m.ALL_Y2)} in the last 12 months — '
               f'{_money(m.AWS_Y2)} on AWS and {_money(m.GCP_Y2)} on GCP — outside both Cloud Services budgets.')
    if len(k['lumpy']):
        top = k['lumpy'].head(3)
        out.append('Largest one-time charges: ' + ', '.join(
            f'{r.Provider} {fr._short(r.Service, 26)} {_money(r.Cost)} ({r.Date:%b %y})' for r in top.itertuples()) + '.')
    k['insights'] = out


# --------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------
def chart_monthly(kd: KPIData) -> go.Figure:
    m = kd.k['monthly']
    fig = go.Figure()
    for p, c in [(GCP, SVC), (AWS, SVC), (GCP, MKT), (AWS, MKT)]:
        if (p, c) in m:
            fig.add_bar(x=m.index, y=m[(p, c)], name=f'{p} {c}', marker=dict(color=COLORS[(p, c)], line=dict(width=0)),
                        hovertemplate='%{y:$,.0f}')
    tax = sum(m[(p, TAX)] for p in (AWS, GCP) if (p, TAX) in m)
    fig.add_bar(x=m.index, y=tax, name='Tax (both)', marker=dict(color=C_TAX, line=dict(width=0)), hovertemplate='%{y:$,.0f}')
    fig.update_layout(barmode='stack')
    fig.add_vline(x=kd.y2[0] - pd.Timedelta(days=15), line=dict(color=AXIS, width=1))
    fig.add_annotation(x=kd.y1[0], y=1.0, yref='paper', text=f'Year 1 · {_yr_label(kd.y1)}', showarrow=False,
                       xanchor='left', font=dict(color=MUTED, size=11))
    fig.add_annotation(x=kd.y2[0], y=1.0, yref='paper', text=f'Year 2 · {_yr_label(kd.y2)}', showarrow=False,
                       xanchor='left', font=dict(color=MUTED, size=11))
    return _layout(fig, 420)


def chart_runrate(kd: KPIData) -> go.Figure:
    svc = kd.k['svc']
    fig = go.Figure()
    for p in (GCP, AWS):
        fig.add_scatter(x=svc.index, y=svc[p], name=f'{p} monthly', mode='markers',
                        marker=dict(color=C_PROVIDER[p], size=7, opacity=0.6), hovertemplate='%{y:$,.0f}')
        fig.add_scatter(x=svc.index, y=svc[p].rolling(3, min_periods=1).mean(), name=f'{p} 3-mo avg', mode='lines',
                        line=dict(color=C_PROVIDER[p], width=2), hovertemplate='%{y:$,.0f}')
        last = svc[p].rolling(3).mean().iloc[-1]
        fig.add_annotation(x=svc.index[-1], y=last, text=f'{p} {_money(last)}', showarrow=False, xanchor='left',
                           xshift=8, font=dict(color=INK, size=11))
    fig.add_hline(y=kd.budget, line=dict(color=INK, width=1.5),
                  annotation=dict(text=f'AWS budget {_money(kd.budget)}', font=dict(color=INK, size=11)),
                  annotation_position='top left')
    fig = _layout(fig, 380)
    fig.update_layout(margin=dict(r=80))
    return fig


def chart_quarterly(kd: KPIData) -> go.Figure:
    svc = kd.k['svc'].copy()
    q = svc.groupby(svc.index.to_period('Q')).sum()
    q = q[svc.groupby(svc.index.to_period('Q')).size() == 3]               # complete quarters only
    fig = go.Figure()
    for p in (GCP, AWS):
        fig.add_bar(x=q.index.astype(str), y=q[p], name=p, marker=dict(color=C_PROVIDER[p], line=dict(width=0)),
                    hovertemplate='%{y:$,.0f}')
    share = q[GCP] / q.sum(axis=1)
    for x, (tot, s) in zip(q.index.astype(str), zip(q.sum(axis=1), share)):
        fig.add_annotation(x=x, y=q.loc[pd.Period(x), [AWS, GCP]].max(), text=f'GCP {s:.0%}', showarrow=False,
                           yshift=12, font=dict(color=INK2, size=11))
    fig.update_layout(barmode='group', bargroupgap=0.08)
    return _layout(fig, 340)


def chart_movers(kd: KPIData, p: str, n: int = 7) -> go.Figure:
    mv = kd.k['movers'][p]['Δ/mo']
    sel = pd.concat([mv.head(n), mv.tail(n)])
    sel = sel[~sel.index.duplicated()]
    sel = sel[sel.abs() >= 1000].sort_values()
    fig = go.Figure(go.Bar(
        y=[fr._short(s, 34) + (' *' if kd.k['movers'][p].loc[s, 'one_time'] else '') for s in sel.index], x=sel, orientation='h',
        marker=dict(color=[BAD if v > 0 else C_PROVIDER[GCP] for v in sel], line=dict(width=0)),
        text=[('+' if v > 0 else '') + _money(v) for v in sel], textposition='outside',
        textfont=dict(color=INK2, size=11), hovertemplate='%{y}: %{x:$,.0f}/mo<extra></extra>'))
    fig = _layout(fig, max(260, 26 * len(sel) + 40), legend=False)
    lim = sel.abs().max() * 1.35
    fig.update_layout(hovermode='closest')
    fig.update_xaxes(range=[-lim, lim], tickformat='$.2~s', showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor=AXIS)
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK2))
    return fig


def chart_genai(kd: KPIData) -> go.Figure:
    g = kd.k['genai']
    fig = go.Figure()
    for p in (GCP, AWS):
        fig.add_scatter(x=g.index, y=g[p], name=p, mode='lines+markers', line=dict(color=C_PROVIDER[p], width=2),
                        marker=dict(size=7), hovertemplate='%{y:$,.0f}')
    return _layout(fig, 300)


def chart_tagcov(kd: KPIData) -> go.Figure:
    t = kd.k['tagcov']
    fig = go.Figure()
    for p in (GCP, AWS):
        s = t[p].dropna()
        fig.add_scatter(x=s.index, y=s, name=p, mode='lines+markers', line=dict(color=C_PROVIDER[p], width=2),
                        marker=dict(size=7), hovertemplate='%{y:.0%}')
    fig.add_hline(y=0.9, line=dict(color=GOOD, width=1.5),
                  annotation=dict(text='Target ≥ 90%', font=dict(color=fr.T['good_text'], size=11)), annotation_position='top left')
    fig = _layout(fig, 280, yfmt='.0%')
    fig.update_yaxes(range=[0, 1])
    return fig


# --------------------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------------------
def _fmt(v, f):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return {'money': _money, 'pct': lambda x: f'{x:.1%}', 'count': lambda x: f'{int(x)} / 12'}[f](v)


def _delta(a, b, f, bad_up):
    if a is None or b is None or any(isinstance(x, float) and np.isnan(x) for x in (a, b)):
        return '<td>—</td>'
    if f == 'money':
        txt, d = (_pct(b / a - 1) if a else '—'), b - a
    elif f == 'pct':
        d = b - a
        txt = f'{d * 100:+.1f} pts'
    else:
        d = b - a
        txt = f'{d:+.0f}'
    cls = '' if bad_up is None or abs(d) < 1e-9 else (' class="bad"' if (d > 0) == bad_up else ' class="good"')
    return f'<td{cls}>{txt}</td>'


def scorecard_html(kd: KPIData) -> str:
    sc = kd.k['scorecard']
    y1, y2 = _yr_label(kd.y1), _yr_label(kd.y2)
    head = ('<tr><th rowspan="2">KPI</th><th colspan="3">AWS</th><th colspan="3">GCP</th><th colspan="3">Combined</th></tr>'
            + '<tr>' + f'<th>Y1</th><th>Y2</th><th>Δ</th>' * 3 + '</tr>')
    body = []
    for r in sc.itertuples():
        cells = [f'<td>{html.escape(r.KPI)}' + (f'<div class="note">{html.escape(r.note)}</div>' if r.note else '') + '</td>']
        for a, b in [(r.AWS_Y1, r.AWS_Y2), (r.GCP_Y1, r.GCP_Y2), (r.ALL_Y1, r.ALL_Y2)]:
            cells += [f'<td>{_fmt(a, r.fmt)}</td>', f'<td><b>{_fmt(b, r.fmt)}</b></td>', _delta(a, b, r.fmt, r.bad_up)]
        body.append('<tr>' + ''.join(cells) + '</tr>')
    return (f'<p class="note">Y1 = {y1} · Y2 = {y2}. Red/green show whether the change is unfavorable/favorable.</p>'
            f'<div class="tbl"><table class="sc"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table></div>')


def build_report(kd: KPIData, out_path: str | None = None) -> str:
    k = kd.k
    sc = k['scorecard'].set_index('KPI')
    out_path = out_path or os.path.join(HERE, f'Cloud_KPI_Report_{kd.end:%Y_%m}.html')
    t, rr = sc.loc['Total spend (all-in)'], sc.loc['Cloud Services run-rate (3-mo avg / month)']
    share = sc.loc['Share of Cloud Services spend']
    aws_budget = sc.loc[f'AWS months within {_money(kd.budget)} budget']

    def tile(label, value, detail, cls=''):
        return (f'<div class="tile"><div class="k">{label}</div><div class="v {cls}">{value}</div>'
                f'<div class="d">{detail}</div></div>')

    g_all = t.ALL_Y2 / t.ALL_Y1 - 1
    tiles = '<div class="tiles">' + ''.join([
        tile('Total cloud spend · last 12 mo', _money(t.ALL_Y2), f'<span class="{"bad" if g_all > 0 else "good"}">{_pct(g_all)}</span> vs prior 12 mo ({_money(t.ALL_Y1)})'),
        tile('Cloud Services run-rate', _money(rr.ALL_Y2) + '/mo', f'was {_money(rr.ALL_Y1)}/mo a year ago'),
        tile(f'<span class="dot" style="background:{C_PROVIDER[AWS]}"></span>AWS run-rate', _money(rr.AWS_Y2) + '/mo',
             f'{_pct(rr.AWS_Y2 / rr.AWS_Y1 - 1)} YoY · {int(aws_budget.AWS_Y2)}/12 months within budget'),
        tile(f'<span class="dot" style="background:{C_PROVIDER[GCP]}"></span>GCP run-rate', _money(rr.GCP_Y2) + '/mo',
             f'<span class="bad">{_pct(rr.GCP_Y2 / rr.GCP_Y1 - 1)}</span> YoY'),
        tile('GCP share of Cloud Services', f'{share.GCP_Y2:.0%}', f'was {share.GCP_Y1:.0%} in the prior 12 months'),
        tile('Marketplace · last 12 mo', _money(sc.loc['Marketplace'].ALL_Y2), f'AWS {_money(sc.loc["Marketplace"].AWS_Y2)} · GCP {_money(sc.loc["Marketplace"].GCP_Y2)}'),
    ]) + '</div>'

    lumpy = k['lumpy'].head(12).copy()
    lumpy['Month'] = lumpy['Date'].dt.strftime('%b %Y')
    lumpy = lumpy.set_index('Service')[['Provider', 'Category', 'Month', 'Cost']]
    lumpy.index = [fr._short(s, 44) for s in lumpy.index]
    lumpy.index.name = 'Service'
    lumpy_html = _table(lumpy, {'Cost': _money})

    def mv_table(p):
        m = k['movers'][p]
        m = pd.concat([m.tail(8)[::-1], m.head(5)]).drop(columns='one_time')
        m.index = [fr._short(s, 40) for s in m.index]
        m.index.name = f'{p} service'
        return _table(m, {c: _money for c in m.columns})

    figs = [chart_monthly(kd), chart_runrate(kd), chart_quarterly(kd), chart_movers(kd, GCP), chart_movers(kd, AWS),
            chart_genai(kd), chart_tagcov(kd)]
    fh = [_fig_html(f, i == 0) for i, f in enumerate(figs)]
    insights = '<ul class="ins a">' + ''.join(f'<li>{html.escape(i)}</li>' for i in k['insights']) + '</ul>'
    extra_css = """
table.sc th{text-align:center}table.sc td{text-align:right}table.sc td:first-child{text-align:left;min-width:220px}
table.sc tr th:nth-child(n){border-bottom:1px solid var(--grid)}
table.sc td:nth-child(4),table.sc td:nth-child(7){border-right:1px solid var(--grid)}
"""
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Cloud KPI Report</title><style>{fr.css()}{extra_css}</style></head>
<body><main>
<header><div><h1>AWS vs GCP — Cloud KPI Report</h1><p class="sub">Trailing 24 months · {_yr_label(kd.y1)} vs {_yr_label(kd.y2)}</p></div>
<div class="meta">Net cost (GCP includes credits) · AWS budget {_money(kd.budget)}/mo<br>Generated {datetime.now():%Y-%m-%d %H:%M}</div></header>
{tiles}
<section><h2>Key findings</h2>{insights}</section>
<section><h2>KPI scorecard</h2>{scorecard_html(kd)}</section>
<section><h2>Monthly spend by cloud and category</h2>
<p class="lede">Cloud Services (first-party) vs Marketplace (third-party software billed through the cloud) and Tax.</p>{fh[0]}</section>
<section><h2>Cloud Services run-rate</h2><p class="lede">Dots are monthly actuals; lines are 3-month averages.</p>{fh[1]}
<h3>By quarter (complete quarters)</h3>{fh[2]}</section>
<section><h2>What changed — service movers (Y2 vs Y1, average $/month)</h2>
<h3>GCP</h3>{fh[3]}<h3>AWS</h3>{fh[4]}<p class="note">* mostly a single one-time charge (&gt;70% of the year in one month).</p>
<details><summary>Mover tables</summary><div class="cols"><div>{mv_table(GCP)}</div><div>{mv_table(AWS)}</div></div></details></section>
<section><h2>GenAI / ML spend</h2><p class="lede">AWS: Bedrock (incl. Claude, Marketplace-billed), SageMaker, Amazon Q, Kendra, Personalize · GCP: Gemini API, Vertex AI and related.</p>{fh[5]}</section>
<section><h2>Allocation readiness — environment tag / label coverage</h2>
<p class="lede">Share of taggable Cloud Services spend that carries an environment tag (AWS, from Jan 2025) or label (GCP).</p>{fh[6]}</section>
<section><h2>One-time and lumpy charges</h2><p class="lede">Single-month charges ≥ $100K and over 3× that service's median month — prepaid commits, annual licenses.</p>{lumpy_html}</section>
<p class="note">Sources: FinOps_AWS/Monthly_Cost_For_*.csv · FinOps_GCP/{GCP_HISTORY} (BigQuery billing export).</p>
</main></body></html>"""
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(doc)
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--month', help='End month YYYY-MM (default: latest month both clouds have)')
    p.add_argument('--budget', type=float, default=AWS_MONTHLY_BUDGET)
    p.add_argument('--refresh-gcp', action='store_true', help='Re-query BigQuery before building')
    p.add_argument('--out')
    p.add_argument('--theme', choices=list(fr.THEMES), default='dark')
    a = p.parse_args(argv)
    set_theme(a.theme)
    if a.refresh_gcp:
        print('GCP history refreshed:', refresh_gcp_history(a.month))
    kd = compute(load_all(), a.month, a.budget)
    print('Report written:', build_report(kd, a.out))
    for line in kd.k['insights']:
        print('  -', line)


if __name__ == '__main__':
    main()
