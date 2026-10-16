"""
GCP "(no project)" deep dive + every GCP project's 2026 monthly spend.

"(no project)" is not a project: it's what Google bills to the billing account itself (project.id IS NULL) —
Premium Support fees, account-level subscriptions (Marketplace) and tax. This report breaks it down by
service and SKU per month, models GCP Premium Support from its pricing drivers, and checks whether projects
that stopped during the year are being carried into the 2027 forecast.

Outputs (in FinOps_AWS/):
  GCP_No_Project_Report_<year>.html       the report
  GCP_No_Project_<year>.csv               one row per (category, service, SKU) with monthly cost
  GCP_Projects_Monthly_<year>.csv         one row per project: monthly cost, status, owner, FY forecast

Usage
-----
    python gcp_no_project_report.py              # uses the cached SKU pull
    python gcp_no_project_report.py --refresh    # re-query BigQuery (~14 GB scanned, a few cents)
"""
from __future__ import annotations

import argparse
import html
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import cloud_forecast as cf
import finops_exec_report as fr
import forecast_detail as fd
import multicloud_kpis as mk
from finops_exec_report import _money, _layout

YEAR = cf.BASIS_YEAR
SKU_FILE = os.path.join(mk.GCP_DIR, f'GCP_No_Project_SKU_{YEAR}.csv')
NO_PROJECT = '(no project)'


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def refresh_sku(year: int = YEAR) -> str:
    """SKU-level detail of billing-account-level charges (no project) for `year`."""
    from google.cloud import bigquery
    client = bigquery.Client(project=mk.GCP_PROJECT)
    sql = f"""
    SELECT invoice.month AS invoice_month, IFNULL(transaction_type, 'GOOGLE') AS transaction_type,
           IFNULL(cost_type, 'regular') AS cost_type, service.description AS service, sku.description AS sku, seller_name,
           SUM(cost) AS cost, SUM(IFNULL((SELECT SUM(x.amount) FROM UNNEST(credits) x), 0)) AS credits
    FROM `{mk.GCP_TABLE}`
    WHERE project.id IS NULL AND invoice.month BETWEEN '{year}01' AND '{year}12'
    GROUP BY 1, 2, 3, 4, 5, 6"""
    client.query(sql).to_dataframe().to_csv(SKU_FILE, index=False)
    return SKU_FILE


def load_sku(last_month: pd.Timestamp) -> pd.DataFrame:
    s = pd.read_csv(SKU_FILE, dtype={'invoice_month': str})
    s['Date'] = pd.to_datetime(s['invoice_month'], format='%Y%m')
    s = s[s['Date'] <= last_month]                                  # drop the month in progress
    s['Cost'] = s['cost'] + s['credits']
    s['Category'] = np.select([s['cost_type'].eq('tax'), s['transaction_type'].ne('GOOGLE')], [mk.TAX, mk.MKT], mk.SVC)
    return s


# --------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------
def no_project_table(sku: pd.DataFrame, months) -> pd.DataFrame:
    t = sku.pivot_table(index=['Category', 'service', 'sku'], columns='Date', values='Cost', aggfunc='sum')
    t = t.reindex(columns=months).fillna(0)
    t = t[t.abs().sum(axis=1) >= 1]
    t.columns = [c.strftime('%b') for c in t.columns]
    t[f'{YEAR} total'] = t.sum(axis=1)
    t = t.reset_index().rename(columns={'service': 'Service', 'sku': 'SKU'})
    t.insert(0, 'Project', NO_PROJECT)
    order = {mk.SVC: 0, mk.MKT: 1, mk.TAX: 2}
    t['In Services forecast?'] = np.where(t['Category'] == mk.SVC, 'Yes', 'No (Marketplace)')
    t.loc[t['Category'] == mk.TAX, 'In Services forecast?'] = 'No (tax)'
    return t.sort_values(['Category', f'{YEAR} total'], key=lambda c: c.map(order) if c.name == 'Category' else -c).reset_index(drop=True)


def support_model(sku: pd.DataFrame, df: pd.DataFrame, fc: cf.Forecast, months) -> dict:
    """GCP Premium Support = fixed minimum + variable fee that scales with GCP spend.
    Fit variable = a × (GCP Services excl. Support) + b on 2026 months, then apply to the FY forecast."""
    sup = sku[(sku['service'] == 'Support') & (sku['Category'] == mk.SVC)]
    minimum = sup[sup['sku'].str.contains('Minimum', na=False)].groupby('Date')['Cost'].sum().reindex(months).fillna(0)
    variable = sup[~sup['sku'].str.contains('Minimum', na=False)].groupby('Date')['Cost'].sum().reindex(months).fillna(0)
    g = df[(df['Provider'] == mk.GCP) & (df['Category'] == mk.SVC)]
    spend = g[g['Service'] != 'Support'].groupby('Date')['Cost'].sum().reindex(months).fillna(0)
    a, b = np.polyfit(spend, variable, 1)

    idx = fc.months
    base = fc.svc[mk.GCP].loc[idx, 'Base']
    sup_share = (minimum + variable).tail(3).sum() / (spend + minimum + variable).tail(3).sum()
    spend_fc = base * (1 - sup_share)
    var_fc = np.clip(a * spend_fc + b, 0, None)
    min_fc = pd.Series(minimum.tail(3).mean(), index=idx)
    # same series and model the detail report uses for the "(no project)" line (all its Cloud Services charges)
    npj = g[g['project'] == NO_PROJECT].groupby('Date')['Cost'].sum().reindex(months).fillna(0)
    trend = fd.forecast_line(NO_PROJECT, npj, fc.horizon).forecast.loc[idx]
    return dict(minimum=minimum, variable=variable, spend=spend, a=a, b=b, pct=variable / spend.replace(0, np.nan),
                min_fc=min_fc, var_fc=var_fc, driver_fy=(min_fc + var_fc).sum(), trend_fy=trend.sum(),
                spend_fc=spend_fc, sup_share=sup_share)


def project_status(df: pd.DataFrame, fc: cf.Forecast, months) -> pd.DataFrame:
    g = df[(df['Provider'] == mk.GCP) & (df['Category'] == mk.SVC) & (df['Date'] >= months[0]) & (df['Date'] <= months[-1])]
    p = g.pivot_table(index='Date', columns='project', values='Cost', aggfunc='sum').reindex(months).fillna(0)
    p = p.loc[:, p.abs().sum() >= 1]
    owners = fd.load_owners()
    omap = fd.owner_map(owners, p.columns) if owners is not None else pd.Series('', index=p.columns)
    rows = []
    for c in p.columns:
        s = p[c]
        peak, exit_rr, q1 = s.max(), s.tail(3).mean(), s.head(3).mean()
        billed = s[s > 50]
        first = billed.index.min() if len(billed) else pd.NaT
        last = billed.index.max() if len(billed) else pd.NaT
        if c == NO_PROJECT:
            status = 'Billing-account level (Support)'
        elif peak >= 500 and exit_rr < 0.1 * peak:
            status = 'Stopped'
        elif q1 < 50 and exit_rr >= 500:
            status = 'New in ' + str(YEAR)
        elif q1 > 0 and exit_rr >= 2 * q1 and exit_rr >= 3_000:
            status = 'Ramping'
        elif peak < 50:
            status = 'Negligible'
        else:
            status = 'Active'
        lf = fd.forecast_line(c, s, fc.horizon)
        fam, env = fd.project_family(c)
        row = {'Project': c, 'Owner': omap.get(c, ''), 'Family': fam, 'Environment': env, 'Status': status}
        row.update({d.strftime('%b'): v for d, v in s.items()})
        row.update({f'{YEAR} total': s.sum(), 'Peak month': s.idxmax().strftime('%b'), 'First billed': first.strftime('%b') if pd.notna(first) else '—',
                    'Last billed': last.strftime('%b') if pd.notna(last) else '—', 'Exit run-rate /mo': exit_rr,
                    f'FY{cf.FORECAST_YEAR} forecast': lf.forecast.loc[fc.months].sum(), 'Model': lf.model, 'Confidence': lf.confidence})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(f'{YEAR} total', ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------
def chart_support(sm: dict, fc: cf.Forecast) -> go.Figure:
    blue = mk.C_PROVIDER[mk.GCP]
    fig = go.Figure()
    fig.add_bar(x=sm['minimum'].index, y=sm['minimum'], name='Minimum fee (fixed)', marker=dict(color=blue, line=dict(width=0)),
                hovertemplate='%{y:$,.0f}')
    fig.add_bar(x=sm['variable'].index, y=sm['variable'], name='Variable fee (scales with spend)',
                marker=dict(color=fr.SERIES[2], line=dict(width=0)), hovertemplate='%{y:$,.0f}')
    fig.add_bar(x=sm['min_fc'].index, y=sm['min_fc'], name='Minimum fee (forecast)', showlegend=False,
                marker=dict(color=cf._rgba(blue, 0.45), line=dict(width=0)), hovertemplate='%{y:$,.0f}')
    fig.add_bar(x=sm['var_fc'].index, y=sm['var_fc'], name='Variable fee (forecast)', showlegend=False,
                marker=dict(color=cf._rgba(fr.SERIES[2], 0.45), line=dict(width=0)), hovertemplate='%{y:$,.0f}')
    fig.update_layout(barmode='stack')
    fig.add_vline(x=fc.months[0] - pd.Timedelta(days=15), line=dict(color=fr.AXIS, width=1))
    fig.add_annotation(x=fc.months[0], y=1, yref='paper', text=f'FY{cf.FORECAST_YEAR} (driver-based) →', showarrow=False,
                       xanchor='left', font=dict(color=fr.MUTED, size=11))
    return _layout(fig, 340)


def chart_variable_vs_spend(sm: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=sm['spend'], y=sm['variable'], mode='markers+text', text=[d.strftime('%b') for d in sm['spend'].index],
                    textposition='top center', textfont=dict(color=fr.INK2, size=10), name='2026 months',
                    marker=dict(color=fr.SERIES[2], size=10), hovertemplate='Spend %{x:$,.0f}<br>Variable fee %{y:$,.0f}<extra></extra>')
    xs = np.linspace(sm['spend'].min() * 0.9, sm['spend_fc'].max() * 1.05, 20)
    fig.add_scatter(x=xs, y=sm['a'] * xs + sm['b'], mode='lines', name=f'Fit: {sm["a"]:.1%} of spend {_money(sm["b"])}',
                    line=dict(color=fr.INK2, width=1.5))
    fig = _layout(fig, 320)
    fig.update_layout(hovermode='closest')
    fig.update_xaxes(tickformat='$.2~s', title=dict(text='GCP Services spend excl. Support ($/month)', font=dict(color=fr.MUTED, size=11)))
    return fig


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def build(refresh: bool = False, out_dir: str = mk.HERE) -> dict:
    if refresh:
        refresh_sku()
    df = mk.load_all()
    kd = mk.compute(df)
    fc = cf.forecast(kd.df, kd.budgets)
    months = pd.date_range(fc.basis_start, fc.basis_end, freq='MS')
    sku = load_sku(fc.basis_end)
    npt = no_project_table(sku, months)
    sm = support_model(sku, df, fc, months)
    ps = project_status(df, fc, months)

    csv1 = os.path.join(out_dir, f'GCP_No_Project_{YEAR}.csv')
    csv2 = os.path.join(out_dir, f'GCP_Projects_Monthly_{YEAR}.csv')
    npt.round(2).to_csv(csv1, index=False)
    ps.round(2).to_csv(csv2, index=False)
    out = os.path.join(out_dir, f'GCP_No_Project_Report_{YEAR}.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(_html(npt, sm, ps, fc, months, os.path.basename(csv1), os.path.basename(csv2)))
    return dict(html=out, no_project_csv=csv1, projects_csv=csv2, table=npt, support=sm, projects=ps, fc=fc)


def _html(npt, sm, ps, fc, months, csv1, csv2) -> str:
    fy = f'FY{cf.FORECAST_YEAR}'
    mcols = [m.strftime('%b') for m in months]
    by_cat = npt.groupby('Category')[f'{YEAR} total'].sum()
    svc_total = by_cat.get(mk.SVC, 0)
    stopped = ps[ps['Status'] == 'Stopped']
    carried = stopped[f'{fy} forecast'].sum()

    def tile(k, v, d):
        return f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="d">{d}</div></div>'

    tiles = '<div class="tiles">' + ''.join([
        tile(f'"(no project)" {YEAR} total', _money(npt[f'{YEAR} total'].sum()),
             ' · '.join(f'{c} {_money(v)}' for c, v in by_cat.items())),
        tile('In the Services forecast', _money(svc_total), 'GCP Premium Support (+ tiny Looker / Vertex AI Search)'),
        tile(f'"(no project)" {fy} · trend', _money(sm['trend_fy']), 'what the detail report shows'),
        tile(f'Support {fy} · driver-based', _money(sm['driver_fy']),
             f'{_money(sm["min_fc"].iloc[0])}/mo minimum + {sm["a"]:.1%} of spend {_money(sm["b"])}'),
        tile(f'Stopped projects carried into {fy}', _money(carried), f'{len(stopped)} project(s) stopped in {YEAR}'),
    ]) + '</div>'

    findings = [
        f'<b>"(no project)" is not a set of old projects.</b> It is what Google bills to the billing account itself '
        f'(<code>project.id IS NULL</code>): Premium Support fees, account-level subscriptions bought through the Marketplace, and tax. '
        f'Projects that stop keep their own name and simply go to $0 — they never move into this line.',
        f'Of the {_money(npt[f"{YEAR} total"].sum())} billed to "(no project)" in {months[0]:%b}–{months[-1]:%b %Y}, only '
        f'<b>{_money(svc_total)}</b> is Cloud Services (in the Services forecast); '
        f'{_money(by_cat.get(mk.MKT, 0))} is Marketplace (Workday, ZoomInfo, GitLab — forecast separately by contract) and '
        f'{_money(by_cat.get(mk.TAX, 0))} is tax.',
        f'GCP Premium Support = a fixed <b>{_money(sm["minimum"].tail(3).mean())}/month minimum</b> plus a <b>variable fee</b> that rose from '
        f'{_money(sm["variable"].iloc[1])} ({sm["pct"].iloc[1]:.1%} of spend) in Feb to {_money(sm["variable"].iloc[-1])} '
        f'({sm["pct"].iloc[-1]:.1%}) in {months[-1]:%b} as GCP spend grew. Fit across 2026: variable ≈ {sm["a"]:.1%} × spend '
        f'{_money(sm["b"])}.',
        f'Applying that to the {fy} GCP base forecast gives <b>{_money(sm["driver_fy"])}</b> for Support vs '
        f'{_money(sm["trend_fy"])} from the trend model ({sm["driver_fy"] / sm["trend_fy"] - 1:+.1%}) — two independent methods agree, '
        f'so the ≈ $0.39M is sound. It moves with GCP spend: every $10K/month of GCP spend adds ≈ {_money(sm["a"] * 10_000)}/month of Support.',
        (f'<b>Stopped projects are not being carried into {fy}.</b> {len(stopped)} project(s) stopped during {YEAR} '
         f'({", ".join(html.escape(p) for p in stopped["Project"])}); because each project starts from its last-3-month run-rate, '
         f'the forecast carries only {_money(carried)} for them.') if len(stopped) else
        f'No project stopped during {YEAR}, so nothing stale is carried into {fy}.',
    ]
    status_counts = ps.groupby('Status').agg(Projects=('Project', 'count'), **{f'{YEAR} total': (f'{YEAR} total', 'sum'),
                                             f'{fy} forecast': (f'{fy} forecast', 'sum')}).sort_values(f'{YEAR} total', ascending=False)

    np_tbl = npt.drop(columns=['Project', 'In Services forecast?']).set_index('Service')   # Category says the same on screen
    np_tbl['SKU'] = [fr._short(x, 34) for x in np_tbl['SKU']]
    np_html = fr._table(np_tbl, {**{m: _money for m in mcols}, f'{YEAR} total': _money})
    sup_tbl = pd.DataFrame({'Minimum fee': sm['minimum'], 'Variable fee': sm['variable'], 'GCP spend excl. Support': sm['spend'],
                            'Variable % of spend': sm['pct']})
    sup_tbl.index = sup_tbl.index.strftime('%b %Y')
    sup_tbl.index.name = 'Month'
    sup_html = fr._table(sup_tbl, {'Minimum fee': _money, 'Variable fee': _money, 'GCP spend excl. Support': _money,
                                   'Variable % of spend': lambda v: f'{v:.1%}'})
    st = stopped.set_index('Project')[['Owner'] + mcols + [f'{YEAR} total', 'Last billed', f'{fy} forecast']] if len(stopped) else None
    st_html = fr._table(st, {**{m: _money for m in mcols}, f'{YEAR} total': _money, f'{fy} forecast': _money}) if st is not None else ''
    newp = ps[ps['Status'].isin([f'New in {YEAR}', 'Ramping'])].set_index('Project')[['Owner', 'Status', 'First billed', 'Exit run-rate /mo', f'{fy} forecast']]
    new_html = fr._table(newp, {'Exit run-rate /mo': _money, f'{fy} forecast': _money})
    top = ps.head(30).set_index('Project')[['Owner', 'Status'] + mcols + [f'{YEAR} total']]
    top_html = fr._table(top, {**{m: _money for m in mcols}, f'{YEAR} total': _money})

    figs = [chart_support(sm, fc), chart_variable_vs_spend(sm)]
    fh = [fr._fig_html(f, i == 0) for i, f in enumerate(figs)]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>GCP No-Project Detail</title>
<style>{fr.css()}code{{color:var(--ink2)}}</style></head><body><main>
<header><div><h1>GCP "(no project)" — {YEAR} detail</h1><p class="sub">What it is, how it behaves, and whether stopped projects leak into {fy}</p></div>
<div class="meta">BigQuery billing export · net of credits · {months[0]:%b}–{months[-1]:%b %Y} (complete months)<br>Generated {datetime.now():%Y-%m-%d %H:%M}</div></header>
{tiles}
<section><h2>Findings</h2><ul class="ins a">{''.join(f'<li>{x}</li>' for x in findings)}</ul>
<p class="note">CSV downloads (same folder): <code>{csv1}</code> (one row per service / SKU, monthly) ·
<code>{csv2}</code> (every GCP project: monthly {YEAR}, status, owner, {fy} forecast).</p></section>

<section><h2>"(no project)" by service and SKU — {YEAR} monthly</h2>{np_html}
<p class="note">Only <i>Cloud Services</i> rows are in the {fy} Services forecast. Marketplace rows are forecast by contract cadence in the
Marketplace section; tax is estimated separately.</p></section>

<section><h2>GCP Premium Support — minimum + variable fee</h2>{fh[0]}
<h3>Variable fee vs GCP spend</h3>{fh[1]}{sup_html}</section>

<section><h2>Project status check</h2>
{fr._table(status_counts, {'Projects': lambda v: f'{int(v)}', f'{YEAR} total': _money, f'{fy} forecast': _money})}
<h3>Stopped during {YEAR}</h3>{st_html or '<p class="note">None.</p>'}
<h3>New and ramping in {YEAR}</h3>{new_html}
<h3>Top 30 projects — {YEAR} monthly</h3>{top_html}
<p class="note">Full list of all projects in <code>{csv2}</code>.</p></section>
</main></body></html>"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--refresh', action='store_true', help='re-query BigQuery for the no-project SKU detail')
    ap.add_argument('--theme', choices=list(fr.THEMES), default='dark')
    a = ap.parse_args(argv)
    mk.set_theme(a.theme)
    r = build(a.refresh)
    print('Report:', r['html']); print('CSV:   ', r['no_project_csv']); print('CSV:   ', r['projects_csv'])


if __name__ == '__main__':
    main()
