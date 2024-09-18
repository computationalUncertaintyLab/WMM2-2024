import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import boto3

def show_cases_over_time():
    
    # Check if dataset is available in session state
    if 'dataset' not in st.session_state:
        AWS_S3_BUCKET = "wmm2-2024"
        AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
        AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]

        s3_client = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        st.session_state.dataset = pd.read_csv(f"s3://{AWS_S3_BUCKET}/wmm_live.csv"
                                               ,storage_options={"key"   : AWS_ACCESS_KEY_ID,"secret": AWS_SECRET_ACCESS_KEY})
    else:
        df = st.session_state.dataset

        # Convert Timestamp to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # Group by date and count the number of cases per day
        df['Date'] = df['Timestamp'].dt.date
        daily_cases = df.groupby('Date').size().reset_index(name='Cases')

        # Create a cumulative sum of cases
        daily_cases['Cumulative Cases'] = daily_cases['Cases'].cumsum()

        # Get the total number of cases
        total_cases = daily_cases['Cumulative Cases'].iloc[-1]

        # Plot the cumulative cases over time using Plotly
        st.markdown(f"### Cumulative Cases Over Time")
        st.markdown(f"##### Total: {total_cases} cases")
        fig_cumulative = go.Figure()
        fig_cumulative.add_trace(go.Scatter(
            x=daily_cases['Date'], y=daily_cases['Cumulative Cases'],
            mode='lines+markers',
            name='Cumulative Cases',
            marker=dict(color='black'),
            text=[f"Total Cases: {cases}" for cases in daily_cases['Cumulative Cases']],
            hoverinfo='text+x'
        ))
        fig_cumulative.update_layout(
            xaxis_title="Date",
            yaxis_title="Cumulative Cases",
            annotations=[{
                'x': daily_cases['Date'].iloc[-1],
                'y': total_cases,
                'xref': 'x', 'yref': 'y',
                'text': f"Total: {total_cases}",
                'showarrow': True,
                'arrowhead': 2,
                'ax': 0, 'ay': -40
            }],
            height=500
        )
        st.plotly_chart(fig_cumulative)

        # Plot the raw daily cases data points using Plotly
        st.markdown(f"### Daily Cases")
        st.markdown(f"##### Total: {total_cases} cases")
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Bar(
            x=daily_cases['Date'], y=daily_cases['Cases'],
            name='Daily Cases',
            marker=dict(color='black'),
            text=[f"Cases: {cases}" for cases in daily_cases['Cases']],
            hoverinfo='text+x'
        ))
        fig_daily.update_layout(
            xaxis_title="Date",
            yaxis_title="Daily Cases",
            annotations=[{
                'x': daily_cases['Date'].iloc[-1],
                'y': daily_cases['Cases'].max(),
                'xref': 'x', 'yref': 'y',
                'text': f"Total: {total_cases}",
                'showarrow': True,
                'arrowhead': 2,
                'ax': 0, 'ay': -40
            }],
            height=500
        )
        st.plotly_chart(fig_daily)

        st.write(daily_cases)

if __name__ == "__main__":
    show_cases_over_time()
