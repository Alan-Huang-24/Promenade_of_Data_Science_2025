# 0.1 導入必要套件
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 設置 Plotly 預設模板
import plotly.io as pio
pio.templates.default = "plotly_white"

# 設置視覺化風格與顏色
REGION_COLORS = {
    '北部': '#ff9999',
    '中部': '#66b3ff',
    '南部': '#99ff99',
    '東部': '#ffcc99',
    '離島': '#c2c2f0'
}
print('環境設置完成 - 使用 Plotly 進行互動式視覺化')
print('視覺化風格與顏色已更新')

# 定義區域劃分常數
REGION_MAPPING = {
    '北部': ['A', 'C', 'F', 'G', 'H', 'J', 'O'],
    '中部': ['B', 'K', 'N', 'M', 'P'],
    '南部': ['D', 'E', 'I', 'Q', 'T', 'X'],
    '東部': ['U', 'V'],
    '離島': ['W', 'Z']
}

def map_county_to_region(county_cd):
    if pd.isna(county_cd): 
        return np.nan
    for r, codes in REGION_MAPPING.items():
        if str(county_cd).strip() in codes: 
            return r
    return np.nan

def map_numeric_county_to_region(county_cd):
    if pd.isna(county_cd): 
        return np.nan
    return NUMERIC_REGION_MAPPING.get(county_cd, np.nan)

print('工具函數定義完成')



# 0.2 導入資料並定義
df_house = pd.read_csv('數據存取\house_own_subset_sumy_rawdata_simulation.csv', encoding='ascii')
df_house['region'] = df_house['county_cd'].apply(map_county_to_region)
df_house['b_area_num'] = pd.to_numeric(df_house['b_area'], errors='coerce')
df_house['b_age_num'] = pd.to_numeric(df_house['b_age'], errors='coerce')
df_house['age_num'] = pd.to_numeric(df_house['age'], errors='coerce')
df_house['own_yr_num'] = pd.to_numeric(df_house['own_yr'], errors='coerce')
df_house['first_own_house_yr_num'] = pd.to_numeric(df_house['first_own_house_yr'], errors='coerce')
print(f'資料載入完成: {len(df_house):,} 筆記錄')



# 1.1 寸土寸金的北部:居住空間的地域差異

# 選取資料
region_area = df_house.dropna(subset=['region', 'b_area_num']).groupby('region')['b_area_num'].agg(
    ['mean', 'median', 'std', 'count']
).reset_index().sort_values('mean', ascending=False)

# 創建互動式長條圖
fig = go.Figure()

fig.add_trace(go.Bar(
    x=region_area['region'],
    y=region_area['mean'],
    marker_color=[REGION_COLORS[r] for r in region_area['region']],
    marker_line_color='black',
    marker_line_width=1.5,
    text=[f'{m:.1f} m²' for m in region_area['mean']],
    textposition='outside',
    textfont=dict(size=12, color='black', family='Arial'),
    hovertemplate='<b>%{x}</b><br>' +
                  '平均面積: %{y:.1f} m²<br>' +
                  '<extra></extra>',
    customdata=np.stack((region_area['median'], region_area['std'], region_area['count']), axis=-1),
    hovertext=['中位數: {:.1f} m²<br>標準差: {:.1f}<br>樣本數: {:,}'.format(
        m, s, int(c)) for m, s, c in zip(region_area['median'], region_area['std'], region_area['count'])]
))

# 創建折線圖
fig.add_trace(go.Scatter(
    x=region_area['region'],
    y=region_area['count'],
    mode='lines+markers+text', 
    name='數據數量',
    line=dict(color='red', width=2),
    text=[f'{c:,}' for c in region_area['count']],  # 顯示樣本數量
    textposition='top right',
    yaxis='y2',
    hovertemplate='<b>%{x}</b><br>' +
                  '樣本數: %{y:,}<br>' +
                  '<extra></extra>'
))

# 更新布局
fig.update_layout(
    title=dict(
        text='<b>寸土寸金的北部:南北居住空間大不同</b>',
        font=dict(size=18, family='Arial'),
        x=0.5,
        xanchor='center'
    ),
    xaxis=dict(
        title='<b>區域</b>',
        title_font=dict(size=14),
        tickfont=dict(size=12)
    ),
    yaxis=dict(
        title='<b>平均建物面積 (m²)</b>',
        title_font=dict(size=14),
        tickfont=dict(size=12),
        range=[region_area['mean'].min()*0.9, region_area['mean'].max()*1.1],
        gridcolor='lightgray',
        gridwidth=0.5
    ),
    yaxis2=dict(
        title='<b>數據數量</b>',
        title_font=dict(size=14),
        tickfont=dict(size=12),
        dtick=100000, 
        range=[region_area['count'].min()*0.8, region_area['count'].max()*1.2],
        overlaying='y', 
        side='right',    
        tickmode='linear'
    ),
    bargap=0.25,
    margin=dict(l=80, r=100, t=70, b=60),
    height=500,
    hovermode='x unified',
    plot_bgcolor='white',
    paper_bgcolor='white',
    showlegend=False,
)

fig.show()



# 1.2 我們如何擁有家?南北市場的冷與熱

# 選取資料
register_labels = {
    'A': '第一次登記', 
    'B': '買賣', 
    'C': '贈與', 
    'D': '繼承', 
    'E': '拍賣', 
    'F': '其他'
}

register_data = df_house.dropna(subset=['region', 'register_reason_group_cd']).copy()
register_summary = register_data.groupby(['region', 'register_reason_group_cd']).size().unstack(fill_value=0)
register_pct = register_summary.div(register_summary.sum(axis=1), axis=0) * 100
register_pct_plot = register_pct.reindex(['北部', '中部', '南部', '東部', '離島'])

# 創建垂直堆疊長條圖
colors_mapping = {
    'A': '#E8F4F8',
    'B': '#FF6B6B',
    'C': '#FFE66D',
    'D': '#C9ADA7',
    'E': '#A8DADC',
    'F': '#457B9D'
}

fig = go.Figure()

for col in register_pct_plot.columns:
    fig.add_trace(go.Bar(
        name=register_labels.get(col, '其他'),
        x=register_pct_plot.index,
        y=register_pct_plot[col],
        marker_color=colors_mapping.get(col, '#999999'),
        marker_line_color='black',
        marker_line_width=0.5,
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      '比例: %{y:.1f}%<br>' +
                      '<extra></extra>'
    ))

# 添加買賣比例標註
if 'B' in register_pct_plot.columns:
    buy_pct = register_pct_plot['B']
    for i, region in enumerate(register_pct_plot.index):
        # 計算標註位置
        y_pos = register_pct_plot.loc[region, :'A'].sum() + (buy_pct[region] / 2) if 'A' in register_pct_plot.columns else buy_pct[region] / 2
        fig.add_annotation(
            x=region,
            y=y_pos,
            text=f'<b>{buy_pct[region]:.1f}%</b><br>(買賣)',
            showarrow=False,
            font=dict(size=11, color='white', family='Arial'),
            bgcolor='rgba(0,0,0,0.3)',
            borderpad=6
        )

# 更新布局
fig.update_layout(
    title=dict(
        text='<b>房屋取得方式的南北差異</b>',
        font=dict(size=18, family='Arial'),
        x=0.5,
        xanchor='center'
    ),
    xaxis=dict(
        title='<b>區域</b>',
        title_font=dict(size=14),
        tickfont=dict(size=12)
    ),
    yaxis=dict(
        title='<b>取得方式比例 (%)</b>',
        title_font=dict(size=14),
        tickfont=dict(size=12),
        gridcolor='lightgray'
    ),
    barmode='stack',
    height=600,
    hovermode='x unified',
    legend=dict(
        title='<b>取得方式</b>',
        orientation='v',
        yanchor='top',
        y=1,
        xanchor='left',
        x=1.02,
        font=dict(size=11)
    ),
    plot_bgcolor='white',
    paper_bgcolor='white'
)

fig.show()



# 1.3 住家或店家?家的另一種可能

# 選取資料
purpose_labels = {
    'A': '住家用', 
    'F': '住商用', 
    'G': '住工用', 
    'L': '國民住宅', 
    'Q': '其他'
}

purpose_data = df_house.dropna(subset=['region', 'purpose_group_cd']).copy()
purpose_data['purpose_label'] = purpose_data['purpose_group_cd'].map(purpose_labels).fillna('其他')
purpose_summary = purpose_data.groupby(['region', 'purpose_label']).size().unstack(fill_value=0)
purpose_pct = purpose_summary.div(purpose_summary.sum(axis=1), axis=0) * 100
purpose_pct_plot = purpose_pct.reindex(['北部', '中部', '南部', '東部', '離島'])

# 創建垂直堆疊長條圖
purpose_colors = {
    '住家用': '#FF6B6B',
    '住商用': '#FFE66D',
    '住工用': '#A8DADC',
    '國民住宅': '#C9ADA7',
    '其他': '#457B9D'
}

fig = go.Figure()

for col in purpose_pct_plot.columns:
    fig.add_trace(go.Bar(
        name=col,
        x=purpose_pct_plot.index,
        y=purpose_pct_plot[col],
        marker_color=purpose_colors.get(col, '#999999'),
        marker_line_color='black',
        marker_line_width=0.5,
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      '比例: %{y:.1f}%<br>' +
                      '<extra></extra>'
    ))

# 增加住家用與住商用的比例標註
if '住家用' in purpose_pct_plot.columns and '住商用' in purpose_pct_plot.columns:
    home_pct = purpose_pct_plot['住家用']
    commercial_pct = purpose_pct_plot['住商用']
    for i, region in enumerate(purpose_pct_plot.index):
        # 計算標註位置
        y_home_pos = home_pct[region] / 2
        fig.add_annotation(
            x=region,
            y=36,
            text=f'<b>{home_pct[region]:.1f}%</b><br>(住家用)',
            showarrow=False,
            font=dict(size=11, color='white', family='Arial'),
            bgcolor='rgba(0,0,0,0.3)',
            borderpad=4
        )
        y_commercial_pos = commercial_pct[region] / 2
        fig.add_annotation(
            x=region,
            y=6,
            text=f'<b>{commercial_pct[region]:.1f}%</b><br>(住商用)',
            showarrow=False,
            yanchor='middle', 
            font=dict(size=11, color='white', family='Arial'),
            bgcolor='rgba(0,0,0,0.3)',
            borderpad=4
        )

# 更新布局
fig.update_layout(
    title=dict(
        text='<b>家的多元功能:各區域房屋用途比較</b>',
        font=dict(size=18, family='Arial'),
        x=0.5,
        xanchor='center'
    ),
    xaxis=dict(
        title='<b>區域</b>',
        title_font=dict(size=14),
        tickfont=dict(size=12),
    ),
    yaxis=dict(
        title='<b>比例 (%)</b>',
        title_font=dict(size=14),
        tickfont=dict(size=12),
        gridcolor='lightgray',
    ),
    barmode='stack',
    height=500,
    hovermode='x unified',
    legend=dict(
        title='<b>房屋用途</b>',
        orientation='v',
        yanchor='top',
        y=1,
        xanchor='left',
        x=1.02,
        font=dict(size=11)
    ),
    
    plot_bgcolor='white',
    paper_bgcolor='white'
)

fig.show()



# 2.1 被房價推遲的歲月：家的起點，越來越晚

# 選取資料
df_house['first_own_house_age_num'] = df_house['age_num'] - (2023 - df_house['first_own_house_yr_num'])
ages = df_house['age_num'].dropna()
first_ages = df_house['first_own_house_age_num'].dropna()

# 創建直方圖
fig = go.Figure()

fig.add_trace(go.Histogram(
    x=first_ages,
    name='初次持有房屋年齡分布',
    marker_color='#4A90E2',
    marker_line_color='white',
    marker_line_width=0.5,
    xbins=dict(              
        start=0,
        end=80,
        size=2
    ),
    hovertemplate='年齡≈%{x:.1f}<br>數量: %{y}<extra></extra>'
))

# 加上平均/中位數參考線
median_age = first_ages.median()
fig.add_vline(x=median_age, line_width=3, line_dash='solid', line_color='black',
              annotation_text=f'中位 {median_age:.1f}', annotation_position='top')

# 更新布局
fig.update_layout(
    title=dict(
        text='<b>初次持有房屋年齡分布</b>',
        font=dict(size=18, family='Arial'),
        x=0.5,
        xanchor='center'
    ),
    xaxis=dict(
        title='<b>年齡</b>',
        range=[0, 80],
        zeroline=False
    ),
    yaxis=dict(
        title='<b>數據數目</b>',
        rangemode='tozero',
        gridcolor='lightgray'
    ),
    bargap=0.05,
    plot_bgcolor='white',
    paper_bgcolor='white',
    showlegend=False
)

fig.show()



# 2.2 1980~2020 每五年統計一次購屋筆數

# 選取資料
yearly_data = df_house.dropna(subset=['first_own_house_yr_num']).copy()
yearly_data['first_own_house_yr_num'] = pd.to_numeric(yearly_data['first_own_house_yr_num'], errors='coerce')
data_1980_2020 = yearly_data[
    (yearly_data['first_own_house_yr_num'] >= 1980) &
    (yearly_data['first_own_house_yr_num'] <= 2020)
]

# 5年為一區間統計
bins = np.arange(1980, 2025, 5)
data_1980_2020['year_group'] = pd.cut(data_1980_2020['first_own_house_yr_num'], bins=bins, right=False)
summary_5yr = data_1980_2020.groupby('year_group').size().reset_index(name='count')
summary_5yr['year_mid'] = summary_5yr['year_group'].apply(lambda x: (x.left + x.right) / 2)

# 計算皮爾森相關係數 
corr_5yr = summary_5yr['year_mid'].corr(summary_5yr['count'], method='pearson')

# 創建折線圖
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=summary_5yr['year_mid'],
    y=summary_5yr['count'],
    mode='lines+markers+text',
    name=f'相關係數 = {corr_5yr:.3f})',
    line=dict(color='#0072B2', width=3),
    marker=dict(size=7, color='#0072B2', line=dict(width=1, color='white')),
    text=[f'{c:,}' for c in summary_5yr['count']],
    textposition='top center',
    hovertemplate='年份區間: %{x:.0f}<br>購屋筆數: %{y:,}<extra></extra>'
))

fig.update_layout(
    title=dict(
        text=f'<b>1980–2020 初次購屋數量（每5年統計）</b>',
        font=dict(size=18, family='Arial'),
        x=0.5,
        xanchor='center'
    ),
    xaxis=dict(
        title='年份',
        range=[1980, 2020],
        dtick=5,
        gridcolor='lightgray'
    ),
    yaxis=dict(
        title='購屋數量（筆）',
        gridcolor='lightgray'
    ),
    height=500,
    plot_bgcolor='white',
    paper_bgcolor='white',
    hovermode='x unified',
    showlegend=True,
    legend=dict(
        x=0,           
        y=1,           
        xanchor='left',
        yanchor='top',
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='black',
        borderwidth=1,
        font=dict(size=11)
    )
)
fig.show()



# 創新與沿伸：市場巨變帶來的影響
# 3.1 2019-2023年各區域購屋面積變化趨勢

# 選取資料(5年的資料)
years = [108, 109, 110, 111, 112]
year_labels = [2019, 2020, 2021, 2022, 2023]
area_timeline = {region: [] for region in ['北部', '中部', '南部', '東部', '離島']}

for year in years:
    df_year = pd.read_csv(f'數據存取\house_own_subset_sumy_rawdata_simulation{year}.csv', encoding='ascii')
    df_year['b_area_num'] = pd.to_numeric(df_year['b_area'], errors='coerce')
    df_year['region'] = df_year['county_cd'].apply(map_county_to_region)
    region_avg = df_year.dropna(subset=['region', 'b_area_num']).groupby('region')['b_area_num'].mean()
    
    for region in area_timeline.keys():
        area_timeline[region].append(region_avg.get(region, np.nan))

# 創建互動式折線圖
fig = go.Figure()

for region, values in area_timeline.items():
    fig.add_trace(go.Scatter(
        x=year_labels,
        y=values,
        mode='lines+markers',
        name=region,
        line=dict(color=REGION_COLORS[region], width=3),
        marker=dict(size=10, symbol='circle', line=dict(width=2, color='white')),
        hovertemplate='<b>' + region + '</b><br>' +
                      '年份: %{x}<br>' +
                      '平均面積: %{y:.1f} m²<br>' +
                      '<extra></extra>'
    ))

# 更新布局
fig.update_layout(
    title=dict(
        text='<b>2019-2023年各區域購屋面積變化趨勢</b>',
        font=dict(size=18, family='Arial'),
        x=0.5,
        xanchor='center'
    ),
    xaxis=dict(
        title='<b>年份</b>',
        title_font=dict(size=14),
        tickfont=dict(size=12),
        tickmode='linear',
        tick0=2019,
        dtick=1,
        gridcolor='lightgray'
    ),
    yaxis=dict(
        title='<b>平均建物面積 (m²)</b>',
        title_font=dict(size=14),
        tickfont=dict(size=12),
        gridcolor='lightgray'
    ),
    height=600,
    hovermode='x unified',
    legend=dict(
        title='<b>區域</b>',
        yanchor='top',
        y=0.99,
        xanchor='right',
        x=0.99,
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='black',
        borderwidth=1,
        font=dict(size=12)
    ),
    plot_bgcolor='white',
    paper_bgcolor='white'
)

fig.show()



# 3.2 112年份中初次持有房屋時間統計

# 選取資料
data_15_24 = yearly_data[
    (yearly_data['first_own_house_yr_num'] >= 2015) &
    (yearly_data['first_own_house_yr_num'] <= 2024)
]

year_summary_15_24 = (
    data_15_24
    .groupby('first_own_house_yr_num')
    .size()
    .reset_index(name='count')
)

# 計算皮爾森相關係數
corr_15_24 = year_summary_15_24['first_own_house_yr_num'].corr(
    year_summary_15_24['count'], method='pearson'
)

# 創建折線圖
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=year_summary_15_24['first_own_house_yr_num'],
    y=year_summary_15_24['count'],
    mode='lines+markers+text',
    name=f'相關係數 = {corr_15_24:.3f})',
    line=dict(color='#0072B2', width=3),
    marker=dict(size=7, color='#0072B2', line=dict(width=1, color='white')),
    text=[f'{c:,}' for c in year_summary_15_24['count']],
    textposition='top center',
    hovertemplate='年份: %{x}<br>筆數: %{y:,}<extra></extra>'
))

# 更新布局
fig.update_layout(
    title=dict(
        text=f'<b>2015–2024 初次購屋數量</b>',
        font=dict(size=18, family='Arial'),
        x=0.5,
        xanchor='center'
    ),
    xaxis=dict(
        title='年份',
        range=[2015, 2024],
        dtick=1,
        gridcolor='lightgray'
    ),
    yaxis=dict(
        title='購屋數量（筆）',
        gridcolor='lightgray'
    ),
    height=500,
    plot_bgcolor='white',
    paper_bgcolor='white',
    hovermode='x unified',
    showlegend=True,
    legend=dict(
        x=0,           
        y=1,           
        xanchor='left',
        yanchor='top',
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='black',
        borderwidth=1,
        font=dict(size=11),
    )
)
fig.show()
