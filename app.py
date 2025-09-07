import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="Solar Power Analysis - Plant 1",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 40px;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 30px;
    }
    .sub-header {
        font-size: 30px;
        color: #1F77B4;
        border-bottom: 2px solid #1F77B4;
        padding-bottom: 10px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1F77B4;
        color: white;
    }
    .plot-container {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .data-info {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load data function with caching
@st.cache_data
def load_data():
    # Plant 1 Generation Data
    df_1_Generation = pd.read_csv("Plant_1_Generation_Data.csv")
    # Plant 1 Weather Sensor Data
    df_1_Weather = pd.read_csv("Plant_1_Weather_Sensor_Data.csv")
    
    # Drop PLANT_ID columns if they exist
    if 'PLANT_ID' in df_1_Generation.columns:
        df_1_Generation.drop('PLANT_ID', axis=1, inplace=True)
    if 'PLANT_ID' in df_1_Weather.columns:
        df_1_Weather.drop('PLANT_ID', axis=1, inplace=True)
    
    # Convert DATE_TIME columns
    df_1_Generation['DATE_TIME'] = pd.to_datetime(df_1_Generation['DATE_TIME'], format='%d-%m-%Y %H:%M')
    df_1_Weather['DATE_TIME'] = pd.to_datetime(df_1_Weather['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')
    
    return df_1_Generation, df_1_Weather

# Load data
df_1_gen, df_1_weather = load_data()

# Sidebar for navigation
with st.sidebar:
    st.title("☀️ Solar Power Analysis")
    st.subheader("Plant 1")
    
    analysis_type = option_menu(
        menu_title="Analysis Type",
        options=["Data Overview", "Generation Data", "Weather Data"],
        icons=["clipboard-data", "lightning-charge", "cloud-sun"],
        default_index=0,
    )

# Main content
st.markdown(f'<h1 class="main-header">Solar Power Analysis - Plant 1</h1>', unsafe_allow_html=True)

# Data Overview
if analysis_type == "Data Overview":
    st.markdown('<h2 class="sub-header">Data Overview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Generation Data Info")
        st.dataframe(df_1_gen.head(10))
        st.write(f"**Shape:** {df_1_gen.shape}")
        
        st.markdown("##### Generation Data Description")
        st.dataframe(df_1_gen.describe())
    
    with col2:
        st.markdown("##### Weather Data Info")
        st.dataframe(df_1_weather.head(10))
        st.write(f"**Shape:** {df_1_weather.shape}")
        
        st.markdown("##### Weather Data Description")
        st.dataframe(df_1_weather.describe())
    
    st.markdown("##### Missing Values")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Generation Data:**")
        missing_gen = df_1_gen.isnull().sum().reset_index()
        missing_gen.columns = ['Column', 'Missing Values']
        st.dataframe(missing_gen)
    
    with col2:
        st.write("**Weather Data:**")
        missing_weather = df_1_weather.isnull().sum().reset_index()
        missing_weather.columns = ['Column', 'Missing Values']
        st.dataframe(missing_weather)

# Generation Data Analysis
elif analysis_type == "Generation Data":
    st.markdown('<h2 class="sub-header">Generation Data Analysis</h2>', unsafe_allow_html=True)
    
    # 1) Daily Production distribution
    st.markdown("#### 1) Daily Production Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df_1_gen['DAILY_YIELD'], bins=50, color='blue', kde=True, ax=ax)
    ax.set_title('Daily Production Distribution')
    ax.set_xlabel('Daily Production (kW)')
    ax.set_ylabel('Density')
    ax.grid(True)
    st.pyplot(fig)
    
    # 2) Daily production density per source
    st.markdown("#### 2) Daily Production Density per Source")
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.boxplot(x=df_1_gen['SOURCE_KEY'], y=df_1_gen['DAILY_YIELD'], palette="Blues", ax=ax)
    ax.tick_params(axis='x', rotation=90)
    ax.set_title('Daily Production Density per Source')
    ax.set_xlabel('Source Key')
    ax.set_ylabel('Daily Production (kW)')
    ax.grid(True, axis='y')
    st.pyplot(fig)
    
    # 3) Average daily ac and dc power
    st.markdown("#### 3) Average Daily AC and DC Power")
    df_avg_power = df_1_gen.groupby(df_1_gen['DATE_TIME'].dt.date)[['AC_POWER','DC_POWER']].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(df_avg_power['DATE_TIME'], df_avg_power['AC_POWER'], label='AC Power', color='blue')
    ax.plot(df_avg_power['DATE_TIME'], df_avg_power['DC_POWER'], label='DC Power', color='green')
    ax.set_title('Average Daily AC and DC Power')
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Power (kW)')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    # 4) Hourly daily production density
    st.markdown("#### 4) Hourly Daily Production Density")
    hourly_yield = df_1_gen.groupby(df_1_gen['DATE_TIME'].dt.hour)['DAILY_YIELD'].mean()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    hourly_yield.plot(kind='bar', color='blue', ax=ax)
    ax.set_title('Hourly Daily Production Density')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Average Daily Production (kW)')
    ax.grid(True, axis='y')
    st.pyplot(fig)
    
    # 5) AC and dc power density per source
    st.markdown("#### 5) AC and DC Power Density per Source")
    grouped = df_1_gen.groupby('SOURCE_KEY').mean(numeric_only=True)[['AC_POWER', 'DC_POWER']]
    
    fig, ax = plt.subplots(figsize=(20, 8))
    grouped.sort_values(by='AC_POWER', ascending=False).plot(kind='bar', stacked=True, ax=ax, color=['blue', 'green'])
    ax.set_title('AC and DC Power Density per Source')
    ax.set_xlabel('Source Key')
    ax.set_ylabel('Average Power (kW)')
    ax.tick_params(axis='x', rotation=90)
    ax.legend(['AC Power', 'DC Power'])
    st.pyplot(fig)
    
    # 6) Total production density among sources
    st.markdown("#### 6) Total Production Distribution Among Sources")
    grouped_total_yield = df_1_gen.groupby('SOURCE_KEY')['TOTAL_YIELD'].last()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(grouped_total_yield, labels=grouped_total_yield.index, autopct='%1.1f%%', startangle=140)
    ax.set_title('Total Production Distribution Among Sources')
    st.pyplot(fig)
    
    # 7) Daily total ac and dc power graph
    st.markdown("#### 7) Daily Total AC and DC Power")
    df_daily = df_1_gen.groupby(df_1_gen['DATE_TIME'].dt.date).sum(numeric_only=True)
    
    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    df_daily['AC_POWER'].plot(ax=ax[0], color='blue')
    df_daily['DC_POWER'].plot(ax=ax[1], color='red')
    ax[0].set_title("Daily Total AC Power")
    ax[1].set_title("Daily Total DC Power")
    
    for axis in ax:
        axis.set_xlabel("Date")
        axis.set_ylabel("Total Power (kW)")
    
    st.pyplot(fig)
    
    # 8) Daily and total production
    st.markdown("#### 8) Daily and Total Production")
    daily_gen = df_1_gen.copy()
    daily_gen['date'] = daily_gen['DATE_TIME'].dt.date
    daily_gen = daily_gen.groupby('date').sum(numeric_only=True)
    
    fig, ax = plt.subplots(ncols=2, figsize=(20, 5))
    daily_gen['DAILY_YIELD'].plot(ax=ax[0], color='navy')
    daily_gen['TOTAL_YIELD'].plot(kind='bar', ax=ax[1], color='navy')
    ax[0].set_title('Daily Production')
    ax[1].set_title('Total Production')
    ax[0].set_ylabel('kW', color='navy', fontsize=17)
    st.pyplot(fig)
    
    # 9) AC power per source
    st.markdown("#### 9) AC Power per Source")
    # Sample data for a few sources to avoid overcrowding
    source_sample = df_1_gen['SOURCE_KEY'].unique()[:3]  # Show only first 3 sources
    filtered_df = df_1_gen[df_1_gen['SOURCE_KEY'].isin(source_sample)]
    
    fig = px.line(filtered_df, x='DATE_TIME', y='AC_POWER', color='SOURCE_KEY', 
                  title='AC Power per Source (Sample)', markers=False, line_shape='linear')
    fig.update_layout(
        xaxis_title="Date-Time",
        yaxis_title="AC Power (kW)",
        legend_title="Source Key",
        xaxis=dict(tickangle=-45)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 10) DC power per source
    st.markdown("#### 10) DC Power per Source")
    fig = px.line(filtered_df, x='DATE_TIME', y='DC_POWER', color='SOURCE_KEY', 
                  title='DC Power per Source (Sample)', markers=False, line_shape='linear')
    fig.update_layout(
        xaxis_title="Date-Time",
        yaxis_title="DC Power (kW)",
        legend_title="Source Key",
        xaxis=dict(tickangle=-45)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 11) Weekly production
    st.markdown("#### 11) Weekly Production")
    weekly_production = df_1_gen.resample('W-Mon', on='DATE_TIME').sum(numeric_only=True)
    
    fig = px.bar(weekly_production, x=weekly_production.index, y="DAILY_YIELD", 
                 title='Weekly Production', 
                 labels={'DAILY_YIELD': 'Total Production (kW)', 'index': 'Date'},
                 color="DAILY_YIELD",  
                 color_continuous_scale="Viridis")
    fig.update_layout(
        plot_bgcolor="white", 
        title_font=dict(size=20),  
        xaxis_title_font=dict(size=14),  
        yaxis_title_font=dict(size=14),
        bargap=0.1  
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 12) Daily yield
    st.markdown("#### 12) Daily Yield")
    df_gen = df_1_gen.groupby('DATE_TIME').sum(numeric_only=True).reset_index()
    df_gen['hour'] = df_gen['DATE_TIME'].dt.hour
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_gen['DATE_TIME'],
        y=df_gen['DAILY_YIELD'],
        mode='lines+markers',
        name='Daily Yield',
        line=dict(color='darkblue', width=3),
        marker=dict(size=8, color=df_gen['DAILY_YIELD'], colorscale='Viridis', showscale=True),
        fill='tozeroy', 
        fillcolor='rgba(100, 150, 250, 0.1)'
    ))
    
    fig.update_layout(
        plot_bgcolor="white",
        title="Daily Yield",
        xaxis_title="Date",
        yaxis_title="Daily Yield (kW)",
        hovermode="x",
        xaxis=dict(tickangle=-45, nticks=20),
        yaxis=dict(gridcolor="lightgrey")
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 13) Daily energy production
    st.markdown("#### 13) Daily Energy Production")
    df_1_gen['date'] = df_1_gen['DATE_TIME'].dt.date
    daily_gen = df_1_gen.groupby('date').sum(numeric_only=True)
    
    fig, ax = plt.subplots(figsize=(15, 6))
    daily_gen['DAILY_YIELD'].plot(ax=ax, color='navy')
    ax.set_title('Daily Energy Production', fontdict={'fontsize': 20})
    ax.set_xlabel('Date', fontdict={'fontsize': 16})
    ax.set_ylabel('Daily Energy Production (kW)', color='navy', fontsize=17)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.4)
    st.pyplot(fig)

# Weather Data Analysis
else:
    st.markdown('<h2 class="sub-header">Weather Data Analysis</h2>', unsafe_allow_html=True)
    
    # 14) Temp trends over time
    st.markdown("#### 14) Temperature Trends Over Time")
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(df_1_weather['DATE_TIME'], df_1_weather['AMBIENT_TEMPERATURE'], label='Ambient Temperature', alpha=0.7)
    ax.plot(df_1_weather['DATE_TIME'], df_1_weather['MODULE_TEMPERATURE'], label='Module Temperature', alpha=0.7)
    ax.set_title('Temperature Trends over Time')
    ax.set_xlabel('Date and Time')
    ax.set_ylabel('Temperature (°C)')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    st.pyplot(fig)
    
    # 15) Daily avg temp and irradiation
    st.markdown("#### 15) Daily Average Temperatures and Irradiation")
    df_1_weather['DAY'] = df_1_weather['DATE_TIME'].dt.date
    daily_avg = df_1_weather.groupby('DAY').mean(numeric_only=True)
    
    fig, ax1 = plt.subplots(figsize=(20, 8))
    ax2 = ax1.twinx()
    
    ax1.plot(daily_avg.index, daily_avg['AMBIENT_TEMPERATURE'], 'g-', label='Avg Ambient Temperature')
    ax1.plot(daily_avg.index, daily_avg['MODULE_TEMPERATURE'], 'b-', label='Avg Module Temperature')
    ax2.plot(daily_avg.index, daily_avg['IRRADIATION'], 'r-', label='Avg Irradiation')
    
    ax1.set_ylabel('Temperature (°C)', color='black')
    ax2.set_ylabel('Irradiation', color='black')
    
    ax1.set_title('Daily Average Temperatures and Irradiation')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    st.pyplot(fig)
    
    # 16) Ambient temp vs irradiation
    st.markdown("#### 16) Ambient Temperature vs. Irradiation")
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.scatter(df_1_weather['AMBIENT_TEMPERATURE'], df_1_weather['IRRADIATION'], alpha=0.5)
    ax.set_title('Ambient Temperature vs. Irradiation')
    ax.set_xlabel('Ambient Temperature (°C)')
    ax.set_ylabel('Irradiation')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    st.pyplot(fig)
    
    # 17) Daily min and max ambient temp
    st.markdown("#### 17) Daily Minimum and Maximum Ambient Temperatures")
    daily_min_max = df_1_weather.groupby('DAY').agg({'AMBIENT_TEMPERATURE': ['min', 'max']})
    
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(daily_min_max.index, daily_min_max['AMBIENT_TEMPERATURE']['min'], label='Min Ambient Temperature', marker='o')
    ax.plot(daily_min_max.index, daily_min_max['AMBIENT_TEMPERATURE']['max'], label='Max Ambient Temperature', marker='o')
    ax.fill_between(daily_min_max.index, daily_min_max['AMBIENT_TEMPERATURE']['min'], 
                   daily_min_max['AMBIENT_TEMPERATURE']['max'], color='grey', alpha=0.1)
    ax.set_title('Daily Minimum and Maximum Ambient Temperatures')
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature (°C)')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    st.pyplot(fig)
    
    # 18) Hourly avg temp and irradiation
    st.markdown("#### 18) Hourly Average Temperatures and Irradiation")
    df_1_weather['HOUR'] = df_1_weather['DATE_TIME'].dt.hour
    hourly_avg = df_1_weather.groupby('HOUR').mean(numeric_only=True)
    
    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax2 = ax1.twinx()
    
    ax1.plot(hourly_avg.index, hourly_avg['AMBIENT_TEMPERATURE'], 'g-', label='Avg Ambient Temperature')
    ax1.plot(hourly_avg.index, hourly_avg['MODULE_TEMPERATURE'], 'b-', label='Avg Module Temperature')
    ax2.plot(hourly_avg.index, hourly_avg['IRRADIATION'], 'r-', label='Avg Irradiation')
    
    ax1.set_ylabel('Temperature (°C)', color='black')
    ax2.set_ylabel('Irradiation', color='black')
    
    ax1.set_title('Hourly Average Temperatures and Irradiation')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.set_xticks(range(0, 24))
    st.pyplot(fig)
    
    # 19) Distribution of irradiation values
    st.markdown("#### 19) Distribution of Irradiation Values")
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.hist(df_1_weather['IRRADIATION'], bins=50, color='orange', alpha=0.7)
    ax.set_title('Distribution of Irradiation Values')
    ax.set_xlabel('Irradiation')
    ax.set_ylabel('Frequency')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    st.pyplot(fig)
    
    # 20) Hourly avg ambient temp by day of week
    st.markdown("#### 20) Hourly Average Ambient Temperature by Day of Week")
    df_1_weather['DAY_OF_WEEK'] = df_1_weather['DATE_TIME'].dt.dayofweek + 1
    pivot_temp = df_1_weather.groupby(['HOUR', 'DAY_OF_WEEK'])['AMBIENT_TEMPERATURE'].mean().unstack()
    
    fig, ax = plt.subplots(figsize=(20, 8))
    sns.heatmap(pivot_temp, cmap="YlGnBu", annot=True, fmt=".1f", 
                cbar_kws={'label': 'Ambient Temperature (°C)'}, ax=ax)
    ax.set_title('Hourly Average Ambient Temperature by Day of Week')
    ax.set_xlabel('Day of Week (1=Monday, 7=Sunday)')
    ax.set_ylabel('Hour of Day')
    st.pyplot(fig)
    
    # 21) Hourly avg irradiation by day of week
    st.markdown("#### 21) Hourly Average Irradiation by Day of Week")
    pivot_irradiation = df_1_weather.groupby(['HOUR', 'DAY_OF_WEEK'])['IRRADIATION'].mean().unstack()
    
    fig, ax = plt.subplots(figsize=(20, 8))
    sns.heatmap(pivot_irradiation, cmap="YlOrRd", annot=True, fmt=".2f", 
                cbar_kws={'label': 'Irradiation'}, ax=ax)
    ax.set_title('Hourly Average Irradiation by Day of Week')
    ax.set_xlabel('Day of Week (1=Monday, 7=Sunday)')
    ax.set_ylabel('Hour of Day')
    st.pyplot(fig)
    
    # 22) Correlation between ambient temp and irradiation
    st.markdown("#### 22) Correlation between Ambient Temperature and Irradiation")
    correlation = df_1_weather[['AMBIENT_TEMPERATURE', 'IRRADIATION']].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation, cmap="coolwarm", annot=True, vmin=-1, vmax=1, ax=ax)
    ax.set_title('Correlation between Ambient Temperature and Irradiation')
    st.pyplot(fig)
    
    # 23) 7 days moving avg for ambient temp and irradiation
    st.markdown("#### 23) 7-day Moving Averages for Ambient Temperature and Irradiation")
    df_1_weather['AMBIENT_TEMPERATURE_MA'] = df_1_weather['AMBIENT_TEMPERATURE'].rolling(window=168).mean()
    df_1_weather['IRRADIATION_MA'] = df_1_weather['IRRADIATION'].rolling(window=168).mean()
    
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(df_1_weather['DATE_TIME'], df_1_weather['AMBIENT_TEMPERATURE_MA'], 
            label='7-day MA Ambient Temperature', color='blue')
    ax.plot(df_1_weather['DATE_TIME'], df_1_weather['IRRADIATION_MA']*50, 
            label='7-day MA Irradiation (scaled)', color='red')
    ax.set_title('7-day Moving Averages for Ambient Temperature and Irradiation')
    ax.set_xlabel('Date and Time')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    st.pyplot(fig)
    
    # 24) Correlation matrix of key variable
    st.markdown("#### 24) Correlation Matrix of Key Variables")
    # Merge data for correlation analysis
    merged_df = pd.merge(df_1_gen, df_1_weather, on='DATE_TIME', how='inner')
    correlation_matrix = merged_df[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 
                                   'DC_POWER', 'AC_POWER', 'DAILY_YIELD']].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.set_title('Correlation Matrix of Key Variables')
    st.pyplot(fig)
    
    # 25) Daily energy production time series
    st.markdown("#### 25) Daily Energy Production Time Series")
    daily_energy_production = df_1_gen.groupby(df_1_gen['DATE_TIME'].dt.date)['DAILY_YIELD'].sum()
    
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(daily_energy_production.index, daily_energy_production.values)
    ax.set_title('Daily Energy Production Time Series')
    ax.set_ylabel('Daily Energy Production (kWh)')
    ax.set_xlabel('Date')
    ax.grid(True)
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("### 📊 Solar Power Generation Analysis Dashboard")
st.markdown("*Explore solar power generation patterns and weather correlations*")
