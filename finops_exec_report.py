"""
AWS FinOps Executive Report
===========================

Builds a self-contained HTML executive report from the monthly Cost Explorer
extracts (Monthly_Cost_For_<MON>_<YEAR>.csv) produced by FinOps.ipynb.

Key ideas
---------
* Every line item is classified as **AWS Services**, **Marketplace** or **Tax**.
  The AWS Services bucket is what is tracked against the monthly budget
  (default $550,000).  Marketplace is lumpy (annual / quarterly commits) and
  hides the real AWS trend when both are mixed together.
* New extracts can carry an authoritative ``BillingEntity`` column
  (``AWS`` / ``AWS Marketplace``) — see ``fetch_month_costs``.  Older CSVs fall
  back to a name-based classifier with explicit overrides below.

Usage
-----
    python finops_exec_report.py                       # latest month, $550K budget
    python finops_exec_report.py --month 2026-09 --budget 550000
    python finops_exec_report.py --include-tax-in-budget

Or from a notebook:
    import finops_exec_report as fr
    data = fr.load_cost_data()
    path = fr.build_report(data, budget=550_000)
"""
from __future__ import annotations

import argparse
import glob
import html
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
MONTHLY_BUDGET = 500_000
MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

AWS, MKT, TAX = 'AWS Services', 'Marketplace', 'Tax'

# Service names that are billed by AWS but don't start with "AWS"/"Amazon".
AWS_SERVICE_OVERRIDES = {
    'EC2 - Other', 'Savings Plans for AWS Compute usage', 'AmazonCloudWatch', 'AmazonWorkMail',
    'CloudWatch Events', 'CodeBuild', 'CodeCatalyst', 'AWSDevOpsAgent', 'Refund',
    'OCBCloudFront', 'OCBLateFee', 'OCBFinance Charge',            # OCB = out-of-cycle bill
    'Training and Certification Vouchers', 'Training Certification Vouchers - backup',
    'Training Certification Vouchers - backup2',
}
# Third-party models on Bedrock are invoiced through AWS Marketplace.
# Flip this to True if you want them counted against the AWS Services budget.
BEDROCK_3P_AS_AWS = False

SP_NAME = 'Savings Plans for AWS Compute usage'
EC2_NAME = 'Amazon Elastic Compute Cloud - Compute'
# Line items that cannot carry a resource tag; excluded from tag-coverage math.
NON_TAGGABLE = {SP_NAME, 'AWS Support (Enterprise)', 'AWS Support (Business)', 'AWS Support (Developer)',
                'AWS Premium Support', 'AWS Cost Explorer', 'AWS Budgets', 'Refund', 'Amazon Registrar',
                'OCBCloudFront', 'OCBLateFee', 'OCBFinance Charge'}
# Friendly short names for charts / narrative.
SHORT_NAMES = {
    'Amazon Relational Database Service': 'RDS', EC2_NAME: 'EC2 Compute', 'Amazon Simple Storage Service': 'S3',
    SP_NAME: 'Savings Plans fee', 'EC2 - Other': 'EC2-Other (EBS/NAT/xfer)', 'Amazon OpenSearch Service': 'OpenSearch',
    'AWS Support (Enterprise)': 'Enterprise Support', 'Amazon CloudFront': 'CloudFront', 'Amazon ElastiCache': 'ElastiCache',
    'AmazonCloudWatch': 'CloudWatch', 'Amazon Virtual Private Cloud': 'VPC', 'Amazon Elastic Load Balancing': 'ELB',
    'Amazon Managed Workflows for Apache Airflow': 'MWAA (Airflow)', 'AWS CloudTrail': 'CloudTrail',
    'Amazon Elastic Container Service': 'ECS', 'Amazon Elastic Container Service for Kubernetes': 'EKS',
    'Amazon Elastic File System': 'EFS', 'Amazon DynamoDB': 'DynamoDB', 'AWS Premium Support': 'Premium Support',
    'Amazon Managed Streaming for Apache Kafka': 'MSK (Kafka)', 'Amazon GuardDuty': 'GuardDuty',
}

# Normalize renamed products so history lines up.
SERVICE_RENAMES = {
    'Datadog Pro': 'Datadog',
    'Apache Kafka® on Confluent Cloud™ - Annual Commits':
        'Apache Kafka® & Apache Flink® on Confluent Cloud™ - Annual Commits',
}

# Short vendor labels for Marketplace charts.
VENDOR_PATTERNS = [
    (r'Confluent', 'Confluent (Kafka/Flink)'), (r'Datadog', 'Datadog'), (r'CrowdStrike', 'CrowdStrike'),
    (r'Grafana Cloud', 'Grafana Cloud'), (r'Kong', 'Kong Konnect'), (r'WIZ', 'Wiz'),
    (r'Red Canary', 'Red Canary'), (r'Upwind', 'Upwind'), (r'Normalyze', 'Normalyze'),
    (r'NetSPI', 'NetSPI'), (r'Proofpoint', 'Proofpoint'), (r'Tines', 'Tines'), (r'dbt Cloud', 'dbt Cloud'),
    (r'CloudFix', 'CloudFix'), (r'Conduktor', 'Conduktor'), (r'Palo Alto', 'Palo Alto Networks'),
    (r'GitLab', 'GitLab'), (r'Aviatrix', 'Aviatrix'), (r'SFTP Gateway', 'SFTP Gateway'),
    (r'Amazon Bedrock Edition|Claude Platform', 'Anthropic Claude (Bedrock)'),
]

# Environment tag hygiene: many spellings of the same environment.
ENV_NORMALIZATION = {
    'prod': 'production', 'production': 'production', 'prd': 'production', 'aardvark-prd': 'production',
    'nonprod': 'non-production', 'dev': 'development', 'development': 'development',
    'stg': 'staging', 'stage': 'staging', 'qa': 'test', 'test': 'test', 'loadtest': 'test',
    'shared': 'shared', 'common': 'shared', 'global': 'shared', 'management': 'shared',
    'sandbox': 'sandbox', 'poc': 'sandbox', 'demo': 'sandbox', '': 'untagged',
}

# Palette (validated default, fixed order) + chrome tokens, per theme.
# Dark steps are the palette's own dark-surface steps, not an inversion of light.
THEMES = {
    'dark': dict(
        series=['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'],
        other='#52514e', tax='#6b6a66', ink='#ffffff', ink2='#c3c2b7', muted='#898781', grid='#2c2c2a',
        axis='#383835', good='#0ca30c', good_text='#0ca30c', warn='#fab219', bad='#d03b3b', bad_text='#e66767',
        under_budget='#256abf', hover_bg='#262624', page='#0d0d0d', surface='#1a1a19',
        border='rgba(255,255,255,.10)', warn_text='#fab219',
        seq=[[0, '#1f2633'], [0.15, '#184f95'], [0.4, '#256abf'], [0.7, '#3987e5'], [1, '#b7d3f6']]),
    'light': dict(
        series=['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'],
        other='#c3c2b7', tax='#898781', ink='#0b0b0b', ink2='#52514e', muted='#898781', grid='#e1e0d9',
        axis='#c3c2b7', good='#0ca30c', good_text='#006300', warn='#fab219', bad='#d03b3b', bad_text='#d03b3b',
        under_budget='#86b6ef', hover_bg='#ffffff', page='#f9f9f7', surface='#fcfcfb',
        border='rgba(11,11,11,.10)', warn_text='#b07800',
        seq=[[0, '#f4f8fe'], [0.15, '#cde2fb'], [0.4, '#86b6ef'], [0.7, '#2a78d6'], [1, '#0d366b']]),
}
THEME = 'dark'
T = dict(THEMES[THEME])


def set_theme(name: str = 'dark') -> None:
    """Switch report + chart colors ('dark' or 'light'). Charts read the module globals at call time."""
    global THEME, C_AWS, C_MKT, C_TAX, SERIES, C_OTHER, INK, INK2, MUTED, GRID, AXIS, GOOD, WARN, BAD, SEQ_BLUE
    THEME = name
    T.clear()
    T.update(THEMES[name])
    SERIES = T['series']
    C_AWS, C_MKT, C_TAX, C_OTHER = SERIES[0], SERIES[1], T['tax'], T['other']
    INK, INK2, MUTED, GRID, AXIS = T['ink'], T['ink2'], T['muted'], T['grid'], T['axis']
    GOOD, WARN, BAD, SEQ_BLUE = T['good'], T['warn'], T['bad'], T['seq']


set_theme(THEME)


# --------------------------------------------------------------------------------------
# Loading & classification
# --------------------------------------------------------------------------------------
def classify_service(service: str, billing_entity: str | None = None) -> str:
    if service == 'Tax':
        return TAX
    if isinstance(billing_entity, str) and billing_entity:
        return MKT if 'Marketplace' in billing_entity else AWS
    if 'Amazon Bedrock Edition' in service or service == 'Claude Platform':
        return AWS if BEDROCK_3P_AS_AWS else MKT
    if service.startswith(('AWS', 'Amazon')) or service in AWS_SERVICE_OVERRIDES:
        return AWS
    return MKT


def vendor_label(service: str) -> str:
    for pat, label in VENDOR_PATTERNS:
        if re.search(pat, service):
            return label
    return service


def normalize_env(raw) -> str | None:
    if not isinstance(raw, str):
        return None                      # tag not captured in this extract (pre-2025 files)
    v = raw.split('$', 1)[-1].strip().lower()
    return ENV_NORMALIZATION.get(v, 'other')


def load_cost_data(folder: str = '.', pattern: str = 'Monthly_Cost_For_*.csv') -> pd.DataFrame:
    """Read every monthly extract into one tidy DataFrame (one row per service/env/month)."""
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise FileNotFoundError(f'No files matching {pattern} in {os.path.abspath(folder)}')
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d = d.drop(columns=[c for c in d.columns if c.startswith('Unnamed')])
        d['SourceFile'] = os.path.basename(f)
        d['FileModified'] = datetime.fromtimestamp(os.path.getmtime(f))
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    df['Month'] = df['Month'].str.upper()
    df['Date'] = pd.to_datetime(dict(year=df['Year'], month=df['Month'].map({m: i + 1 for i, m in enumerate(MONTHS)}), day=1))
    df['Service'] = df['Service'].replace(SERVICE_RENAMES)
    if 'BillingEntity' not in df.columns:
        df['BillingEntity'] = None
    df['Category'] = [classify_service(s, b) for s, b in zip(df['Service'], df['BillingEntity'])]
    df['Vendor'] = np.where(df['Category'] == MKT, df['Service'].map(vendor_label), df['Service'])
    env_col = 'Enviroment' if 'Enviroment' in df.columns else 'Environment'
    df['Env'] = df[env_col].map(normalize_env) if env_col in df.columns else None
    df['EnvRaw'] = df[env_col].astype('string').str.split('$', n=1).str[-1] if env_col in df.columns else None

    dupes = df.groupby('Date')['SourceFile'].nunique()
    if (dupes > 1).any():
        raise ValueError(f'More than one file per month: {dupes[dupes > 1].index.strftime("%Y-%m").tolist()}')
    return df


# --------------------------------------------------------------------------------------
# Optional: pull a month straight from Cost Explorer WITH billing entity
# --------------------------------------------------------------------------------------
def fetch_month_costs(profile: dict, year: int, month: int) -> pd.DataFrame:
    """Same output as FinOps.ipynb's extract, plus an authoritative BillingEntity column.

    ``profile`` is one entry of ``aws_profiles`` from FinOps.ipynb
    ({'name','account','key','access','token'}).  Cost Explorer's End date is
    *exclusive*, so the period is [1st of month, 1st of next month).
    """
    import boto3

    start = date(year, month, 1)
    end = start + relativedelta(months=1)
    ce = boto3.client('ce', region_name='us-east-1', aws_access_key_id=profile['key'],
                      aws_secret_access_key=profile['access'], aws_session_token=profile['token'])

    def query(group_by):
        rows, token = [], None
        while True:
            kw = dict(TimePeriod={'Start': start.isoformat(), 'End': end.isoformat()},
                      Granularity='MONTHLY', Metrics=['NetUnblendedCost'], GroupBy=group_by)
            if token:
                kw['NextPageToken'] = token
            resp = ce.get_cost_and_usage(**kw)
            for r in resp['ResultsByTime']:
                for g in r['Groups']:
                    rows.append((*g['Keys'], float(g['Metrics']['NetUnblendedCost']['Amount'])))
            token = resp.get('NextPageToken')
            if not token:
                return rows

    by_env = query([{'Type': 'DIMENSION', 'Key': 'SERVICE'}, {'Type': 'TAG', 'Key': 'Environment'}])
    by_entity = query([{'Type': 'DIMENSION', 'Key': 'SERVICE'}, {'Type': 'DIMENSION', 'Key': 'BILLING_ENTITY'}])
    entity = {svc: ent for svc, ent, _ in by_entity}

    out = pd.DataFrame(by_env, columns=['Service', 'Enviroment', 'Cost'])
    out.insert(0, 'Month', MONTHS[month - 1])
    out.insert(1, 'Year', year)
    out.insert(2, 'Name', profile['name'])
    out.insert(3, 'Account', profile['account'])
    out['BillingEntity'] = out['Service'].map(entity)
    return out


# --------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------
@dataclass
class ReportData:
    df: pd.DataFrame
    month: pd.Timestamp
    budget: float
    include_tax: bool = False
    m: dict = field(default_factory=dict)   # computed metrics


def _avg(s: pd.Series, start, end) -> float:
    return float(s.loc[start:end].mean()) if len(s.loc[start:end]) else 0.0


def analyze(df: pd.DataFrame, month: str | None = None, budget: float = MONTHLY_BUDGET,
            include_tax_in_budget: bool = False) -> ReportData:
    month_ts = pd.Timestamp(month) if month else df['Date'].max()
    df = df[df['Date'] <= month_ts].copy()
    rd = ReportData(df=df, month=month_ts, budget=budget, include_tax=include_tax_in_budget)
    m = rd.m

    cat = df.pivot_table(index='Date', columns='Category', values='Cost', aggfunc='sum').reindex(columns=[AWS, MKT, TAX]).fillna(0)
    cat['Total'] = cat.sum(axis=1)
    cat['Budgeted'] = cat[AWS] + (cat[TAX] if include_tax_in_budget else 0)
    m['cat'] = cat
    budgeted = cat['Budgeted']

    # Month & run-rate
    prev, ly = month_ts - relativedelta(months=1), month_ts - relativedelta(years=1)
    m['cur'] = budgeted.loc[month_ts]
    m['prev'] = budgeted.get(prev, np.nan)
    m['ly'] = budgeted.get(ly, np.nan)
    m['run_rate'] = budgeted.loc[:month_ts].tail(3).mean()
    m['rolling3'] = budgeted.rolling(3).mean()

    # Partial-month detection (extract pulled before the month closed)
    src = df[df['Date'] == month_ts]
    pulled = src['FileModified'].max()
    m['partial'] = bool(pulled < (month_ts + relativedelta(months=1)))
    m['pulled'] = pulled

    # Calendar-year budget tracking
    yr = month_ts.year
    ytd = budgeted.loc[f'{yr}-01-01':month_ts]
    months_elapsed, months_left = len(ytd), 12 - len(ytd)
    m['ytd'] = ytd.sum()
    m['ytd_budget'] = budget * months_elapsed
    m['projected_year'] = m['ytd'] + m['run_rate'] * months_left
    m['annual_budget'] = budget * 12
    m['months_over'] = int((ytd > budget).sum())
    m['ytd_months'] = months_elapsed
    m['headroom_monthly'] = (m['annual_budget'] - m['ytd']) / months_left if months_left else 0.0

    # The journey: peak, trough, rebound of AWS Services
    aws = cat[AWS]
    r3 = aws.rolling(3).mean()
    m['aws_peak_date'], m['aws_peak'] = r3.idxmax(), r3.max()          # 3-mo average, end month
    yr_avg = aws.groupby(aws.index.year).agg(['mean', 'count'])
    m['peak_year'] = int(yr_avg[yr_avg['count'] >= 6]['mean'].idxmax())
    trough_window = aws.loc[m['aws_peak_date']:]
    m['aws_trough_date'], m['aws_trough'] = trough_window.idxmin(), trough_window.min()
    m['aws_cur'] = aws.loc[month_ts]

    # Annual summary (monthly averages so partial years are comparable)
    yearly = cat.groupby(cat.index.year).agg(['sum', 'mean', 'count'])
    ann = pd.DataFrame({
        'Months': yearly[(AWS, 'count')],
        'AWS Services': yearly[(AWS, 'sum')], 'AWS avg / month': yearly[(AWS, 'mean')],
        'Marketplace': yearly[(MKT, 'sum')], 'Tax': yearly[(TAX, 'sum')], 'Total': yearly[('Total', 'sum')],
    })
    ann['AWS avg YoY'] = ann['AWS avg / month'].pct_change()
    ann['Mkt share'] = ann['Marketplace'] / ann['Total']
    m['annual'] = ann

    # Service-level movers (AWS Services only)
    svc = df[df['Category'] == AWS].pivot_table(index='Date', columns='Service', values='Cost', aggfunc='sum').fillna(0)
    m['svc'] = svc
    last3 = (month_ts - relativedelta(months=2), month_ts)
    prior3 = (month_ts - relativedelta(months=5), month_ts - relativedelta(months=3))
    ly3 = (last3[0] - relativedelta(years=1), last3[1] - relativedelta(years=1))
    peak_yr = (pd.Timestamp(f'{m["peak_year"]}-01-01'), pd.Timestamp(f'{m["peak_year"]}-12-01'))
    mv = pd.DataFrame({
        'Current month': svc.loc[month_ts],
        'Last 3 mo avg': svc.loc[last3[0]:last3[1]].mean(),
        'Prior 3 mo avg': svc.loc[prior3[0]:prior3[1]].mean(),
        'Same 3 mo last yr': svc.loc[ly3[0]:ly3[1]].mean(),
        'Peak-year avg': svc.loc[peak_yr[0]:peak_yr[1]].mean(),
        'Trough month': svc.loc[m['aws_trough_date']],
    }).fillna(0)
    mv['Δ vs prior 3 mo'] = mv['Last 3 mo avg'] - mv['Prior 3 mo avg']
    mv['Δ vs last yr'] = mv['Last 3 mo avg'] - mv['Same 3 mo last yr']
    mv['Δ vs peak yr'] = mv['Last 3 mo avg'] - mv['Peak-year avg']
    mv['Δ since trough'] = mv['Current month'] - mv['Trough month']
    mv['% vs last yr'] = mv['Δ vs last yr'] / mv['Same 3 mo last yr'].replace(0, np.nan)
    m['movers'] = mv.sort_values('Last 3 mo avg', ascending=False)

    # Commitments: Savings Plans vs on-demand EC2 compute
    sp = svc.get(SP_NAME, pd.Series(0, index=svc.index))
    ec2 = svc.get(EC2_NAME, pd.Series(0, index=svc.index))
    m['eff_compute'] = sp + ec2
    m['sp'] = sp
    m['ec2'] = ec2
    m['sp_gap_months'] = [d for d in sp.loc[month_ts - relativedelta(months=11):month_ts].index if sp.loc[d] <= 0]
    support = svc.filter(like='Support').sum(axis=1)
    m['support_pct'] = support.loc[last3[0]:last3[1]].sum() / aws.loc[last3[0]:last3[1]].sum()

    # Marketplace
    mk = df[df['Category'] == MKT].pivot_table(index='Date', columns='Vendor', values='Cost', aggfunc='sum').fillna(0)
    m['mkt'] = mk
    t12 = (month_ts - relativedelta(months=11), month_ts)
    p12 = (t12[0] - relativedelta(years=1), t12[1] - relativedelta(years=1))
    mkv = pd.DataFrame({'Trailing 12 mo': mk.loc[t12[0]:t12[1]].sum(), 'Prior 12 mo': mk.loc[p12[0]:p12[1]].sum()})
    mkv['Δ'] = mkv['Trailing 12 mo'] - mkv['Prior 12 mo']
    last_charge = mk.apply(lambda s: s[s > 1].index.max() if (s > 1).any() else pd.NaT)
    mkv['Last charge'] = last_charge
    m['mkt_vendors'] = mkv[(mkv['Trailing 12 mo'].abs() > 1) | (mkv['Prior 12 mo'].abs() > 1)].sort_values('Trailing 12 mo', ascending=False)

    # Expected Marketplace charges next 6 months (assumes every contract renews at the same amount/cadence)
    rows = []
    for i in range(1, 7):
        fut = month_ts + relativedelta(months=i)
        base = fut - relativedelta(years=1)
        if base in mk.index:
            for vendor, amt in mk.loc[base].items():
                if amt >= 1000:
                    rows.append({'Expected month': fut, 'Vendor': vendor, 'Expected amount': amt, 'Based on': base})
    m['renewals'] = pd.DataFrame(rows)

    # Tag governance (only months where the Environment tag was captured)
    tagged = df[(df['Category'] == AWS) & df['Env'].notna() & ~df['Service'].isin(NON_TAGGABLE)]
    if len(tagged):
        cov = tagged.assign(untagged=tagged['Env'].eq('untagged') * tagged['Cost']).groupby('Date')[['untagged', 'Cost']].sum()
        m['untagged_pct'] = cov['untagged'] / cov['Cost']
        cur_tag = tagged[tagged['Date'] == month_ts]
        m['env_mix'] = cur_tag.groupby('Env')['Cost'].sum().sort_values(ascending=False)
        raw = cur_tag[cur_tag['Env'] != 'untagged'].groupby('Env')['EnvRaw'].apply(lambda s: sorted(set(s.dropna())))
        m['env_variants'] = raw[raw.map(len) > 1]
        m['untagged_svc'] = (cur_tag[cur_tag['Env'] == 'untagged'].groupby('Service')['Cost'].sum()
                             .sort_values(ascending=False).head(10))
    else:
        m['untagged_pct'] = pd.Series(dtype=float)

    # GenAI
    genai = df[df['Service'].str.contains('Bedrock|Claude|SageMaker|Amazon Q|Kendra|Personalize', regex=True)]
    m['genai'] = genai.pivot_table(index='Date', values='Cost', aggfunc='sum')['Cost']

    _build_insights(rd)
    return rd


def _money(x: float, short: bool = True) -> str:
    if pd.isna(x):
        return '—'
    sign = '-' if x < 0 else ''
    x = abs(x)
    if short and x >= 1e6:
        return f'{sign}${x / 1e6:,.2f}M'
    if short and x >= 1e4:
        return f'{sign}${x / 1e3:,.0f}K'
    if short and x >= 1e3:
        return f'{sign}${x / 1e3:,.1f}K'
    return f'{sign}${x:,.0f}'


def _pct(x: float, signed: bool = True) -> str:
    if pd.isna(x) or np.isinf(x):
        return '—'
    return f'{x:+.1%}' if signed else f'{x:.1%}'


def _short(name: str, n: int = 38) -> str:
    name = SHORT_NAMES.get(name, name.replace('Amazon ', '').replace('AWS ', ''))
    return name if len(name) <= n else name[: n - 1] + '…'


def _build_insights(rd: ReportData) -> None:
    m, b = rd.m, rd.budget
    mname = rd.month.strftime('%B %Y')
    wins, watch, actions = [], [], []

    # Journey
    drop = m['aws_peak'] - m['aws_cur']
    wins.append(f'AWS Services spend is {_money(m["aws_cur"])} in {mname}, {_money(drop)}/month ({drop / m["aws_peak"]:.0%}) '
                f'below the peak 3-month average of {_money(m["aws_peak"])}/month (ending {m["aws_peak_date"]:%b %Y}).')
    best = m['movers'].sort_values('Δ vs peak yr').head(4)
    wins.append('Biggest structural reductions vs ' + str(m['peak_year']) + ' average: ' +
                ', '.join(f'{_short(s, 26)} ({_money(v)}/mo)' for s, v in best['Δ vs peak yr'].items()) + '.')

    # Budget
    if m['cur'] <= b:
        wins.append(f'{mname} is {_money(b - m["cur"])} under the {_money(b)} monthly AWS budget '
                    f'({m["cur"] / b:.0%} utilized).')
    else:
        watch.append(f'{mname} is {_money(m["cur"] - b)} OVER the {_money(b)} monthly AWS budget.')
    proj, ann = m['projected_year'], m['annual_budget']
    (wins if proj <= ann else watch).append(
        f'{rd.month.year} projected AWS Services spend at the current 3-month run-rate is {_money(proj)} '
        f'vs {_money(ann)} annual budget ({_money(ann - proj)} {"headroom" if proj <= ann else "overrun"}).')

    # Rebound since trough
    if m['aws_trough_date'] < rd.month and m['aws_cur'] > m['aws_trough'] * 1.05:
        reb = m['aws_cur'] - m['aws_trough']
        drivers = m['movers'].drop(index=SP_NAME, errors='ignore').sort_values('Δ since trough', ascending=False).head(4)
        watch.append(f'AWS Services has rebounded {_money(reb)}/month ({reb / m["aws_trough"]:.0%}) since the '
                     f'{m["aws_trough_date"]:%b %Y} low of {_money(m["aws_trough"])}. Main drivers: ' +
                     ', '.join(f'{_short(s, 26)} (+{_money(v)})' for s, v in drivers['Δ since trough'].items()) + '.'
                     + (' Part of the low is artificial: no Savings Plans fee was billed that month.'
                        if m['sp'].get(m['aws_trough_date'], 1) <= 0 else ''))

    # Fast-growing services
    mv = m['movers']
    growing = mv[(mv.index != SP_NAME) & (mv['Δ vs prior 3 mo'] > 2_000) & (mv['Δ vs prior 3 mo'] / mv['Prior 3 mo avg'].replace(0, np.nan) > 0.15)]
    for s, r in growing.sort_values('Δ vs prior 3 mo', ascending=False).head(5).iterrows():
        watch.append(f'{_short(s, 40)} is growing: {_money(r["Last 3 mo avg"])}/mo over the last 3 months vs '
                     f'{_money(r["Prior 3 mo avg"])}/mo the 3 months before.')

    # Commitments
    eff = m['eff_compute']
    gap_avg = eff[m['sp_gap_months']].mean() if m['sp_gap_months'] else np.nan
    now_avg = eff.loc[rd.month - relativedelta(months=2):rd.month].mean()
    if m['sp_gap_months'] and now_avg > gap_avg * 1.15:
        actions.append(f'Effective compute (EC2 Compute + Savings Plans fee) is {_money(now_avg)}/mo now vs {_money(gap_avg)}/mo '
                       'in the months with no Savings Plans fee. Check Savings Plans utilization in Cost Explorer — '
                       'an under-utilized commitment costs more than on-demand.')
    if m['sp_gap_months']:
        actions.append('Savings Plans charges were $0 in ' + ', '.join(d.strftime('%b %Y') for d in m['sp_gap_months']) +
                       ' — check for lapsed/renewed commitments and on-demand exposure in those months.')
    actions.append(f'AWS Support is {m["support_pct"]:.1%} of AWS Services spend (last 3 months). '
                   'Validate Enterprise Support tier vs. Business/Enterprise On-Ramp at the current spend level.')

    # Tagging
    up = m['untagged_pct']
    if len(up):
        cur_unt = up.loc[rd.month] if rd.month in up.index else up.iloc[-1]
        actions.append(f'{cur_unt:.0%} of AWS Services spend has no Environment tag. Top untagged: ' +
                       ', '.join(f'{_short(s, 24)} ({_money(v)})' for s, v in m['untagged_svc'].head(4).items()) +
                       '. Enforce tagging (Tag Policies / SCP) and activate the cost-allocation tag.')
        if len(m['env_variants']):
            actions.append('Environment tag values are inconsistent — ' +
                           '; '.join(f'{k}: {", ".join(v)}' for k, v in m['env_variants'].items()) +
                           '. Standardize allowed values via an AWS Organizations Tag Policy.')

    # Marketplace
    ren = m['renewals']
    if len(ren):
        nxt = ren[ren['Expected month'] <= rd.month + relativedelta(months=3)]
        if len(nxt):
            actions.append('Expected Marketplace charges in the next 3 months: ' +
                           ', '.join(f'{v} {_money(a)} ({d:%b})' for d, v, a in zip(nxt['Expected month'], nxt['Vendor'], nxt['Expected amount'])) +
                           '. Review renewals 60–90 days ahead for right-sizing / negotiation.')
    mk12 = m['mkt_vendors']['Trailing 12 mo'].sum()
    watch.append(f'Marketplace spend over the trailing 12 months is {_money(mk12)} '
                 f'({_money(mk12 / 12)}/month on average), outside the AWS Services budget.')

    if m['partial']:
        watch.insert(0, f'⚠ {mname} extract was pulled on {m["pulled"]:%Y-%m-%d}, before month-end — numbers are partial.')

    m['wins'], m['watch'], m['actions'] = wins, watch, actions


# --------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------
def _layout(fig: go.Figure, height: int = 380, yfmt: str = '$.2~s', legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=16, t=8, b=8), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='system-ui, -apple-system, Segoe UI, sans-serif', size=12, color=INK2),
        hoverlabel=dict(bgcolor=T['hover_bg'], bordercolor=GRID, font=dict(color=INK)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0, title=None, traceorder='normal') if legend else None,
        showlegend=legend, bargap=0.25, hovermode='x unified',
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickcolor=AXIS, ticks='outside', tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=True, zerolinecolor=AXIS, tickformat=yfmt, tickfont=dict(color=MUTED))
    return fig


def _budget_line(fig, budget, label='Monthly budget'):
    fig.add_hline(y=budget, line=dict(color=INK, width=1.5),
                  annotation=dict(text=f'{label} {_money(budget)}', font=dict(color=INK, size=11), xanchor='left', x=0),
                  annotation_position='top left')


def chart_journey(rd: ReportData) -> go.Figure:
    cat = rd.m['cat']
    fig = go.Figure()
    for col, color in [(AWS, C_AWS), (MKT, C_MKT), (TAX, C_TAX)]:
        fig.add_bar(x=cat.index, y=cat[col], name=col, marker=dict(color=color, line=dict(width=0)),
                    hovertemplate='%{y:$,.0f}')
    fig.update_layout(barmode='stack')
    _budget_line(fig, rd.budget, 'AWS budget')
    return _layout(fig, 420)


def chart_aws_trend(rd: ReportData) -> go.Figure:
    cat, m = rd.m['cat'], rd.m
    fig = go.Figure()
    fig.add_bar(x=cat.index, y=cat['Budgeted'], name='AWS Services (monthly)',
                marker=dict(color=[T['under_budget'] if v <= rd.budget else BAD for v in cat['Budgeted']], line=dict(width=0)),
                hovertemplate='%{y:$,.0f}')
    fig.add_scatter(x=cat.index, y=m['rolling3'], name='3-month average', mode='lines',
                    line=dict(color=C_AWS, width=2), hovertemplate='%{y:$,.0f}')
    for d, lbl in [(m['aws_peak_date'], 'Peak 3-mo'), (m['aws_trough_date'], 'Low'), (rd.month, 'Now')]:
        val = m['rolling3'].loc[d] if lbl.startswith('Peak') else cat.loc[d, 'Budgeted']
        fig.add_annotation(x=d, y=val, text=f'{lbl}<br>{_money(val)}',
                           showarrow=True, arrowhead=0, arrowcolor=MUTED, ay=-34, font=dict(size=11, color=INK))
    _budget_line(fig, rd.budget)
    return _layout(fig, 400)


def chart_budget_ytd(rd: ReportData) -> go.Figure:
    yr = rd.month.year
    s = rd.m['cat']['Budgeted'].loc[f'{yr}-01-01':rd.month]
    var = rd.budget - s
    fig = go.Figure()
    fig.add_bar(x=s.index.strftime('%b'), y=s, name='Actual', marker=dict(color=C_AWS, line=dict(width=0)),
                text=[f'{_money(v)}' for v in s], textposition='outside', textfont=dict(color=INK2, size=11),
                customdata=var, hovertemplate='Actual %{y:$,.0f}<br>Under budget by %{customdata:$,.0f}<extra></extra>')
    cum = var.cumsum()
    _budget_line(fig, rd.budget)
    fig.update_yaxes(range=[0, max(rd.budget, s.max()) * 1.15])
    fig.add_annotation(xref='paper', yref='paper', x=1, y=1.12, showarrow=False, xanchor='right',
                       text=f'Cumulative YTD {"headroom" if cum.iloc[-1] >= 0 else "overrun"}: <b>{_money(abs(cum.iloc[-1]))}</b>',
                       font=dict(color=INK, size=12))
    return _layout(fig, 340, legend=False)


def chart_top_services(rd: ReportData, n: int = 7, months: int = 24) -> go.Figure:
    svc = rd.m['svc'].loc[rd.month - relativedelta(months=months - 1):rd.month]
    top = svc.loc[rd.month - relativedelta(months=2):].mean().nlargest(n).index
    other = svc.drop(columns=top).sum(axis=1)
    fig = go.Figure()
    for i, s in enumerate(top):
        fig.add_bar(x=svc.index, y=svc[s], name=_short(s, 30), marker=dict(color=SERIES[i], line=dict(width=0)),
                    hovertemplate='%{y:$,.0f}')
    fig.add_bar(x=svc.index, y=other, name='All other AWS', marker=dict(color=C_OTHER, line=dict(width=0)),
                hovertemplate='%{y:$,.0f}')
    fig.update_layout(barmode='stack')
    return _layout(fig, 420)


def chart_movers(rd: ReportData, col: str, n: int = 8, title_up='Increase', title_down='Decrease') -> go.Figure:
    mv = rd.m['movers'][col]
    sel = pd.concat([mv.nlargest(n), mv.nsmallest(n)])
    sel = sel[sel.abs() > 500].sort_values()
    fig = go.Figure(go.Bar(
        y=[_short(s, 34) for s in sel.index], x=sel, orientation='h',
        marker=dict(color=[BAD if v > 0 else C_AWS for v in sel], line=dict(width=0)),
        text=[('+' if v > 0 else '') + _money(v) for v in sel], textposition='outside',
        textfont=dict(color=INK2, size=11), hovertemplate='%{y}: %{x:$,.0f}/mo<extra></extra>'))
    fig = _layout(fig, max(300, 26 * len(sel) + 40), legend=False)
    fig.update_layout(hovermode='closest')
    fig.update_xaxes(tickformat='$.2~s', gridcolor=GRID, showgrid=True, zeroline=True, zerolinecolor=AXIS)
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK2))
    lim = sel.abs().max() * 1.35
    fig.update_xaxes(range=[-lim, lim])
    return fig


def chart_commitments(rd: ReportData) -> go.Figure:
    m = rd.m
    start = rd.month - relativedelta(months=23)
    fig = go.Figure()
    fig.add_scatter(x=m['ec2'].loc[start:].index, y=m['ec2'].loc[start:], name='EC2 Compute (not covered by SP)',
                    mode='lines+markers', line=dict(color=SERIES[0], width=2), marker=dict(size=7), hovertemplate='%{y:$,.0f}')
    fig.add_scatter(x=m['sp'].loc[start:].index, y=m['sp'].loc[start:], name='Savings Plans fee',
                    mode='lines+markers', line=dict(color=SERIES[1], width=2), marker=dict(size=7), hovertemplate='%{y:$,.0f}')
    fig.add_scatter(x=m['eff_compute'].loc[start:].index, y=m['eff_compute'].loc[start:], name='Effective compute (EC2 + SP fee)',
                    mode='lines+markers', line=dict(color=SERIES[2], width=2), marker=dict(size=7), hovertemplate='%{y:$,.0f}')
    return _layout(fig, 320)


def chart_marketplace_heat(rd: ReportData, months: int = 24) -> go.Figure:
    mk = rd.m['mkt'].loc[rd.month - relativedelta(months=months - 1):rd.month]
    mk = mk.loc[:, mk.sum() > 1000]
    mk = mk[mk.sum().sort_values(ascending=True).index]
    z = mk.T.values
    fig = go.Figure(go.Heatmap(
        z=z, x=mk.index.strftime('%b %y'), y=mk.columns, colorscale=SEQ_BLUE, xgap=2, ygap=2,
        hovertemplate='%{y}<br>%{x}: %{z:$,.0f}<extra></extra>', colorbar=dict(tickformat='$.2~s', thickness=10, outlinewidth=0)))
    fig = _layout(fig, max(260, 30 * len(mk.columns) + 60), legend=False)
    fig.update_layout(hovermode='closest')
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK2))
    return fig


def chart_tagging(rd: ReportData) -> go.Figure:
    up = rd.m['untagged_pct']
    fig = go.Figure(go.Scatter(x=up.index, y=up, mode='lines+markers', name='Untagged share',
                               line=dict(color=BAD, width=2), marker=dict(size=7), hovertemplate='%{y:.1%}<extra></extra>'))
    fig.add_hline(y=0.10, line=dict(color=GOOD, width=1.5),
                  annotation=dict(text='Target ≤ 10%', font=dict(color=T['good_text'], size=11)), annotation_position='top left')
    fig = _layout(fig, 280, yfmt='.0%', legend=False)
    fig.update_yaxes(range=[0, 1])
    return fig


# --------------------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------------------
_CSS_TEMPLATE = """
:root{{--page:{page};--surface:{surface};--ink:{ink};--ink2:{ink2};--muted:{muted};--grid:{grid};
--border:{border};--good:{good_text};--bad:{bad_text};--warn:{warn_text};--aws:{s0};--mkt:{s1};--tax:{tax};color-scheme:{scheme}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--page);color:var(--ink);
font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:32px 16px 64px}}
header{{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-end;gap:12px;margin-bottom:24px}}
h1{{font-size:28px;margin:0;letter-spacing:-.01em}}h2{{font-size:19px;margin:0 0 4px}}h3{{font-size:15px;margin:18px 0 6px}}
.sub{{color:var(--ink2);margin:2px 0 0}}.meta{{color:var(--muted);font-size:12px;text-align:right}}
section{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px 22px;margin:16px 0}}
.lede{{color:var(--ink2);margin:0 0 12px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:16px 0}}
.tile{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 16px}}
.tile .k{{color:var(--ink2);font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
.tile .v{{font-size:28px;font-weight:650;margin:2px 0}}.tile .d{{font-size:12.5px;color:var(--ink2)}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}}
.good{{color:var(--good)}}.bad{{color:var(--bad)}}.warn{{color:var(--warn)}}
.cols{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}}
ul.ins{{margin:6px 0 0;padding-left:0;list-style:none}}ul.ins li{{padding:7px 0 7px 26px;position:relative;border-top:1px solid var(--grid)}}
ul.ins li:first-child{{border-top:0}}ul.ins li::before{{position:absolute;left:2px;top:6px;font-weight:700}}
.w li::before{{content:"✓";color:var(--good)}}.x li::before{{content:"!";color:var(--bad);left:7px}}.a li::before{{content:"→";color:var(--aws)}}
.tbl{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}}
th,td{{padding:6px 10px;text-align:right;border-bottom:1px solid var(--grid);white-space:nowrap}}
th{{color:var(--ink2);font-weight:600;font-size:12px}}th:first-child,td:first-child{{text-align:left;white-space:normal}}
tr.total td{{font-weight:650;border-top:1px solid var(--muted)}}
details{{margin-top:10px}}summary{{cursor:pointer;color:var(--ink2);font-size:13px}}
.note{{color:var(--muted);font-size:12px;margin-top:8px}}
@media print{{body{{background:var(--page)}}section,.tile{{break-inside:avoid}}}}
"""


def css() -> str:
    """Page CSS for the current theme."""
    return _CSS_TEMPLATE.format(s0=SERIES[0], s1=SERIES[1], scheme=THEME, **{k: v for k, v in T.items() if isinstance(v, str)})



def _table(df: pd.DataFrame, fmt: dict, total_row: bool = False) -> str:
    head = ''.join(f'<th>{html.escape(str(c))}</th>' for c in [df.index.name or ''] + list(df.columns))
    body = []
    for i, (idx, row) in enumerate(df.iterrows()):
        cls = ' class="total"' if total_row and i == len(df) - 1 else ''
        cells = [f'<td>{html.escape(str(idx))}</td>']
        for c in df.columns:
            v = row[c]
            f = fmt.get(c, str)
            txt = f(v) if not (isinstance(v, float) and pd.isna(v)) else '—'
            color = ''
            if c.startswith(('Δ', '%')) or 'YoY' in c or c.startswith('Variance'):
                if isinstance(v, (int, float)) and not pd.isna(v) and abs(v) >= (0.0005 if c.startswith('%') or 'YoY' in c else 0.5):
                    up_bad = not c.startswith('Variance')
                    color = ' class="bad"' if (v > 0) == up_bad else ' class="good"'
            cells.append(f'<td{color}>{txt}</td>')
        body.append(f'<tr{cls}>' + ''.join(cells) + '</tr>')
    return f'<div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _fig_html(fig: go.Figure, first: bool) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=True if first else False,
                       config={'displaylogo': False, 'responsive': True,
                               'modeBarButtonsToRemove': ['lasso2d', 'select2d']})


def build_report(rd: ReportData, out_path: str | None = None) -> str:
    m, b = rd.m, rd.budget
    mname = rd.month.strftime('%B %Y')
    out_path = out_path or f'AWS_Executive_Report_{rd.month:%Y_%m}.html'
    cat = m['cat']
    cur_total = cat.loc[rd.month, 'Total']
    var = b - m['cur']
    mom = m['cur'] / m['prev'] - 1 if m['prev'] else np.nan
    yoy = m['cur'] / m['ly'] - 1 if m['ly'] else np.nan
    budget_label = 'AWS Services' + (' + Tax' if rd.include_tax else '')

    tiles = f"""
<div class="tiles">
 <div class="tile"><div class="k"><span class="dot" style="background:var(--aws)"></span>{budget_label} · {rd.month:%b %Y}</div>
  <div class="v">{_money(m['cur'])}</div>
  <div class="d"><span class="{'good' if var >= 0 else 'bad'}">{'▼' if var >= 0 else '▲'} {_money(abs(var))} {'under' if var >= 0 else 'over'} budget</span> · {m['cur'] / b:.0%} of {_money(b)}</div></div>
 <div class="tile"><div class="k">Change</div><div class="v {'bad' if mom > 0 else 'good'}">{_pct(mom)}</div>
  <div class="d">vs prior month · <span class="{'bad' if yoy > 0 else 'good'}">{_pct(yoy)}</span> vs {rd.month - relativedelta(years=1):%b %Y}</div></div>
 <div class="tile"><div class="k">{rd.month.year} YTD vs budget</div><div class="v">{_money(m['ytd'])}</div>
  <div class="d"><span class="{'good' if m['ytd'] <= m['ytd_budget'] else 'bad'}">{_money(abs(m['ytd_budget'] - m['ytd']))} {'under' if m['ytd'] <= m['ytd_budget'] else 'over'}</span> {m['ytd_months']}-month budget of {_money(m['ytd_budget'])}</div></div>
 <div class="tile"><div class="k">{rd.month.year} projected (run-rate)</div><div class="v">{_money(m['projected_year'])}</div>
  <div class="d">3-mo avg {_money(m['run_rate'])}/mo · annual budget {_money(m['annual_budget'])}</div></div>
 <div class="tile"><div class="k"><span class="dot" style="background:var(--mkt)"></span>Marketplace · {rd.month:%b %Y}</div><div class="v">{_money(cat.loc[rd.month, MKT])}</div>
  <div class="d">Trailing 12 mo {_money(m['mkt_vendors']['Trailing 12 mo'].sum())}</div></div>
 <div class="tile"><div class="k">Total AWS invoice · {rd.month:%b %Y}</div><div class="v">{_money(cur_total)}</div>
  <div class="d">incl. Tax {_money(cat.loc[rd.month, TAX])}</div></div>
</div>"""

    def ul(items, cls):
        return f'<ul class="ins {cls}">' + ''.join(f'<li>{html.escape(i)}</li>' for i in items) + '</ul>'

    # Tables
    ann = m['annual'].copy()
    ann.index = [f'{y}{" (partial)" if n < 12 else ""}' for y, n in zip(ann.index, ann['Months'])]
    ann.index.name = 'Year'
    ann_html = _table(ann.drop(columns=['Months']), {
        'AWS Services': _money, 'AWS avg / month': _money, 'Marketplace': _money, 'Tax': _money, 'Total': _money,
        'AWS avg YoY': _pct, 'Mkt share': lambda v: _pct(v, False)})

    mv = m['movers'].head(20)[['Current month', 'Last 3 mo avg', 'Prior 3 mo avg', 'Δ vs prior 3 mo',
                               'Same 3 mo last yr', 'Δ vs last yr', '% vs last yr', 'Peak-year avg', 'Δ vs peak yr']]
    mv = mv.rename(columns={'Peak-year avg': f'{m["peak_year"]} avg', 'Δ vs peak yr': f'Δ vs {m["peak_year"]}'})
    mv.index = [_short(s, 44) for s in mv.index]
    mv.index.name = 'Top 20 AWS services (by last 3 mo)'
    mv_html = _table(mv, {c: _money for c in mv.columns} | {'% vs last yr': _pct})

    mkv = m['mkt_vendors'].copy()
    mkv['Last charge'] = mkv['Last charge'].map(lambda d: d.strftime('%b %Y') if pd.notna(d) else '—')
    mkv.index.name = 'Vendor'
    tot = pd.DataFrame([[mkv['Trailing 12 mo'].sum(), mkv['Prior 12 mo'].sum(), mkv['Δ'].sum(), '']],
                       columns=mkv.columns, index=['Total'])
    mkv_html = _table(pd.concat([mkv, tot]).rename_axis('Vendor'),
                      {'Trailing 12 mo': _money, 'Prior 12 mo': _money, 'Δ': _money}, total_row=True)

    ren = m['renewals']
    if len(ren):
        r = ren.copy()
        r['Expected month'] = r['Expected month'].dt.strftime('%b %Y')
        r['Based on'] = r['Based on'].dt.strftime('%b %Y')
        r = r.set_index('Expected month')
        ren_html = _table(r, {'Expected amount': _money})
        ren_total = f'<p class="note">Total expected over the next 6 months: <b>{_money(ren["Expected amount"].sum())}</b>. ' \
                    'Assumes each contract recurs at the same amount and cadence as 12 months earlier.</p>'
    else:
        ren_html, ren_total = '<p class="note">No recurring Marketplace charges found.</p>', ''

    # Budget table for current year
    yr = rd.month.year
    s = cat.loc[f'{yr}-01-01':rd.month]
    bt = pd.DataFrame({budget_label: s['Budgeted'], 'Budget': b, 'Variance (under)': b - s['Budgeted'],
                       'Cumulative variance': (b - s['Budgeted']).cumsum(), 'Marketplace': s[MKT], 'Tax': s[TAX], 'Total invoice': s['Total']})
    bt.index = bt.index.strftime('%b %Y')
    bt.index.name = 'Month'
    bt.loc['YTD'] = [bt[budget_label].sum(), b * len(s), bt['Variance (under)'].sum(), bt['Variance (under)'].sum(),
                     bt['Marketplace'].sum(), bt['Tax'].sum(), bt['Total invoice'].sum()]
    bt = bt.rename(columns={'Variance (under)': 'Variance', 'Cumulative variance': 'Variance (cumulative)'})
    bt_html = _table(bt, {c: _money for c in bt.columns}, total_row=True)

    env_html = ''
    if 'env_mix' in m:
        em = m['env_mix'][m['env_mix'].abs() >= 1].to_frame('Cost')
        em['Share'] = em['Cost'] / em['Cost'].sum()
        em.index.name = f'Environment ({rd.month:%b %Y})'
        env_html = _table(em, {'Cost': _money, 'Share': lambda v: _pct(v, False)})
        unt = m['untagged_svc'].to_frame('Untagged cost')
        unt.index = [_short(i, 40) for i in unt.index]
        unt.index.name = 'Largest untagged services'
        env_html += '<h3>Where the untagged spend is</h3>' + _table(unt, {'Untagged cost': _money})

    cls = rd.df.groupby(['Category', 'Service'])['Cost'].sum().reset_index()
    cls = cls[cls['Category'] == MKT].sort_values('Cost', ascending=False).set_index('Service')[['Cost']]
    cls.index.name = 'Services classified as Marketplace (all history)'
    cls_html = _table(cls, {'Cost': _money})

    genai = m['genai'].loc[rd.month - relativedelta(months=5):rd.month]

    figs = [chart_journey(rd), chart_aws_trend(rd), chart_budget_ytd(rd), chart_top_services(rd),
            chart_movers(rd, 'Δ vs prior 3 mo'), chart_movers(rd, f'Δ vs peak yr'),
            chart_commitments(rd), chart_marketplace_heat(rd)]
    tag_fig = chart_tagging(rd) if len(m['untagged_pct']) else None
    fh = [_fig_html(f, i == 0) for i, f in enumerate(figs)]
    tag_html = _fig_html(tag_fig, False) if tag_fig else '<p class="note">No Environment tag data captured.</p>'

    partial = (f'<p class="bad"><b>⚠ Partial month:</b> the {mname} extract was pulled on {m["pulled"]:%Y-%m-%d}, before '
               'month-end. Re-pull after the 3rd of next month for final numbers.</p>') if m['partial'] else ''

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>AWS Executive Report</title><style>{css()}</style></head>
<body><main>
<header><div><h1>AWS Executive Cost Report</h1><p class="sub">{mname} · AWS Services budget {_money(b)}/month</p></div>
<div class="meta">Data: {rd.df['Date'].min():%b %Y} – {rd.month:%b %Y} · Net unblended cost<br>Generated {datetime.now():%Y-%m-%d %H:%M}</div></header>
{partial}
{tiles}

<section><h2>Executive summary</h2>
<div class="cols"><div><h3 class="good">What's working</h3>{ul(m['wins'], 'w')}</div>
<div><h3 class="bad">Watch list</h3>{ul(m['watch'], 'x')}</div></div>
<h3>Recommended actions</h3>{ul(m['actions'], 'a')}</section>

<section><h2>The AWS journey — total invoice by category</h2>
<p class="lede">Marketplace commits (orange) arrive as large quarterly/annual charges and create the spikes in the total bill.
Separating them shows the true AWS Services trend against the {_money(b)} budget.</p>{fh[0]}
<h3>Year over year</h3>{ann_html}
<p class="note">"AWS avg / month" normalizes partial years so they can be compared. 2021 starts in October.</p></section>

<section><h2>AWS Services vs budget</h2>
<p class="lede">Monthly AWS Services spend (Marketplace and Tax excluded) with the 3-month rolling average. Red bars exceed budget.</p>{fh[1]}
<h3>{yr} budget tracker</h3>{fh[2]}{bt_html}</section>

<section><h2>What AWS Services are made of</h2>
<p class="lede">Top 7 services over the last 24 months; everything else is grouped.</p>{fh[3]}
<h3>Recent movers: last 3 months vs the 3 months before ($/month)</h3>{fh[4]}
<h3>Structural change: last 3 months vs {m['peak_year']} average ($/month)</h3>{fh[5]}
<details><summary>Service detail table</summary>{mv_html}</details></section>

<section><h2>Commitments & rate optimization</h2>
<p class="lede">EC2 Compute that is not covered by Savings Plans is billed on-demand. Months where the Savings Plans fee drops to $0 indicate a coverage gap.</p>{fh[6]}</section>

<section><h2>Marketplace (outside the AWS Services budget)</h2>
<p class="lede">Charges by vendor, last 24 months. {'Brighter' if THEME == 'dark' else 'Darker'} = larger charge.</p>{fh[7]}
<div class="cols"><div><h3>Vendors — trailing 12 months vs prior 12 months</h3>{mkv_html}</div>
<div><h3>Expected Marketplace charges — next 6 months</h3>{ren_html}{ren_total}</div></div>
<details><summary>Classification check</summary>{cls_html}
<p class="note">Edit AWS_SERVICE_OVERRIDES / BEDROCK_3P_AS_AWS in finops_exec_report.py if anything is misclassified.
New extracts with a BillingEntity column are classified authoritatively.</p></details></section>

<section><h2>Tagging & governance</h2>
<p class="lede">Share of AWS Services spend without an Environment tag (data available from Jan 2025).</p>{tag_html}
<div class="cols"><div>{env_html}</div><div><h3>GenAI / ML spend (last 6 months)</h3>
{_table(genai.to_frame('Cost').set_axis(genai.index.strftime('%b %Y')).rename_axis('Month'), {'Cost': _money})}
<p class="note">Bedrock (incl. Claude models billed via Marketplace), SageMaker, Amazon Q, Kendra, Personalize.</p></div></div></section>

<p class="note">Source files: Monthly_Cost_For_*.csv ({rd.df['SourceFile'].nunique()} months). Budget compares {budget_label} only.</p>
</main></body></html>"""
    with open(out_path, 'w', encoding='utf-8') as fh_:
        fh_.write(doc)
    return out_path


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--folder', default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument('--month', help='Report month YYYY-MM (default: latest available)')
    p.add_argument('--budget', type=float, default=MONTHLY_BUDGET)
    p.add_argument('--include-tax-in-budget', action='store_true')
    p.add_argument('--theme', choices=list(THEMES), default='dark')
    p.add_argument('--out', help='Output HTML path')
    a = p.parse_args(argv)
    set_theme(a.theme)

    df = load_cost_data(a.folder)
    rd = analyze(df, a.month, a.budget, a.include_tax_in_budget)
    out = build_report(rd, a.out or os.path.join(a.folder, f'AWS_Executive_Report_{rd.month:%Y_%m}.html'))
    print(f'Report written: {out}')
    for k in ('wins', 'watch', 'actions'):
        print(f'\n{k.upper()}:')
        for line in rd.m[k]:
            print(f'  - {line}')


if __name__ == '__main__':
    main()
