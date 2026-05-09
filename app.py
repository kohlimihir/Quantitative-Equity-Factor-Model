import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from pathlib import Path

st.set_page_config(
    page_title="Equity Factor Model Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_predictions():
    """Load model predictions"""
    try:
        df = pd.read_csv('data/ensemble_optimize_predictions.csv')
        df['date'] = pd.to_datetime(df['date'])
        # Rename ensemble_pred to prediction for consistency
        if 'ensemble_pred' in df.columns:
            df['prediction'] = df['ensemble_pred']
        return df
    except Exception as e:
        st.error(f"Error loading predictions: {str(e)}")
        return None

@st.cache_data
def load_shap_values():
    """Load SHAP values for explainability"""
    try:
        df = pd.read_csv('data/shap_values.csv')
        return df
    except Exception as e:
        return None

@st.cache_data
def load_performance_metrics():
    """Load performance reports"""
    try:
        oot = pd.read_csv('reports/oot_validation_report.csv')
        turnover = pd.read_csv('reports/turnover_optimized.csv')
        return oot, turnover
    except Exception as e:
        return None, None

@st.cache_data
def load_feature_importance():
    """Load feature importance"""
    try:
        df = pd.read_csv('reports/feature_ic_summary.csv')
        return df
    except Exception as e:
        return None

def main():
    st.title("📈 Equity Factor Model Dashboard")
    st.markdown("**Multi-Factor Quantitative Strategy with Machine Learning**")
    
    predictions = load_predictions()
    shap_values = load_shap_values()
    oot_metrics, turnover_data = load_performance_metrics()
    feature_importance = load_feature_importance()
    
    if predictions is None:
        st.error("⚠️ No data found. Please run the model pipeline first: `python run_all.py --config optimized_low_turnover`")
        return
    
    tabs = st.tabs([
        "📊 Overview", 
        "💼 Portfolio",
        "🎯 Stock Analysis", 
        "🔍 Model Explainability",
        "📈 Performance",
        "⚙️ Configuration"
    ])
    
    with tabs[0]:
        show_overview(predictions, oot_metrics, turnover_data)
    
    with tabs[1]:
        show_portfolio(predictions, oot_metrics, turnover_data)
    
    with tabs[2]:
        show_stock_analysis(predictions, shap_values)
    
    with tabs[3]:
        show_explainability(shap_values, feature_importance)
    
    with tabs[4]:
        show_performance(predictions, oot_metrics, turnover_data)
    
    with tabs[5]:
        show_configuration()

def show_portfolio(predictions, oot_metrics, turnover_data):
    """Portfolio overview and composition"""
    st.header("Portfolio Overview")
    
    # Get latest portfolio
    latest_date = predictions['date'].max()
    latest_preds = predictions[predictions['date'] == latest_date].copy()
    
    # Portfolio construction: Top N per sector
    top_n_per_sector = 4  # Adjust based on your config
    portfolio = latest_preds.groupby('sector').apply(
        lambda x: x.nlargest(top_n_per_sector, 'prediction')
    ).reset_index(drop=True)
    
    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Portfolio Size", f"{len(portfolio)} stocks")
    
    with col2:
        st.metric("Sectors", f"{portfolio['sector'].nunique()}")
    
    with col3:
        avg_pred = portfolio['prediction'].mean()
        st.metric("Avg Prediction", f"{avg_pred:.4f}")
    
    with col4:
        if 'actual' in portfolio.columns:
            avg_actual = portfolio['actual'].mean()
            st.metric("Avg Actual Return", f"{avg_actual:.4f}")
        else:
            st.metric("Avg Actual Return", "N/A")
    
    with col5:
        if oot_metrics is not None and 'OOT_Sharpe' in oot_metrics.columns:
            sharpe = oot_metrics['OOT_Sharpe'].iloc[0]
            st.metric("Sharpe Ratio", f"{sharpe:.3f}")
        else:
            st.metric("Sharpe Ratio", "N/A")
    
    st.markdown("---")
    
    # Portfolio composition
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sector Allocation")
        sector_counts = portfolio['sector'].value_counts()
        
        fig = px.pie(
            values=sector_counts.values,
            names=sector_counts.index,
            title="Portfolio by Sector",
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Prediction Distribution")
        
        fig = px.box(
            portfolio,
            x='sector',
            y='prediction',
            title="Predictions by Sector",
            color='sector'
        )
        fig.update_layout(showlegend=False, xaxis_title="Sector", yaxis_title="Prediction")
        st.plotly_chart(fig, use_container_width=True)
    
    # Portfolio holdings table
    st.subheader("Current Portfolio Holdings")
    
    # Add rank within sector
    portfolio['sector_rank'] = portfolio.groupby('sector')['prediction'].rank(ascending=False, method='dense').astype(int)
    
    # Display table
    display_cols = ['ticker', 'sector', 'prediction', 'sector_rank']
    if 'actual' in portfolio.columns:
        display_cols.append('actual')
    
    portfolio_display = portfolio[display_cols].sort_values(['sector', 'sector_rank'])
    portfolio_display['prediction'] = portfolio_display['prediction'].apply(lambda x: f"{x:.4f}")
    if 'actual' in portfolio_display.columns:
        portfolio_display['actual'] = portfolio_display['actual'].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(portfolio_display, hide_index=True, use_container_width=True)
    
    # Sector breakdown
    st.subheader("Sector Breakdown")
    
    sector_stats = portfolio.groupby('sector').agg({
        'prediction': ['count', 'mean', 'std', 'min', 'max']
    }).round(4)
    sector_stats.columns = ['Count', 'Mean Pred', 'Std Pred', 'Min Pred', 'Max Pred']
    sector_stats = sector_stats.reset_index()
    
    st.dataframe(sector_stats, hide_index=True, use_container_width=True)
    
    # Historical portfolio performance
    st.subheader("Historical Portfolio Performance")
    
    # Calculate portfolio returns over time
    portfolio_history = []
    for date in sorted(predictions['date'].unique()):
        date_preds = predictions[predictions['date'] == date]
        date_portfolio = date_preds.groupby('sector').apply(
            lambda x: x.nlargest(top_n_per_sector, 'prediction')
        ).reset_index(drop=True)
        
        if 'actual' in date_portfolio.columns:
            portfolio_history.append({
                'date': date,
                'avg_prediction': date_portfolio['prediction'].mean(),
                'avg_actual': date_portfolio['actual'].mean(),
                'portfolio_size': len(date_portfolio)
            })
    
    if portfolio_history:
        hist_df = pd.DataFrame(portfolio_history)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_df['date'],
            y=hist_df['avg_prediction'],
            mode='lines',
            name='Avg Prediction',
            line=dict(color='blue', width=2)
        ))
        
        if 'avg_actual' in hist_df.columns:
            fig.add_trace(go.Scatter(
                x=hist_df['date'],
                y=hist_df['avg_actual'],
                mode='lines',
                name='Avg Actual Return',
                line=dict(color='green', width=2)
            ))
        
        fig.update_layout(
            title="Portfolio Average Prediction vs Actual Returns",
            xaxis_title="Date",
            yaxis_title="Return",
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

def show_overview(predictions, oot_metrics, turnover_data):
    """Enhanced overview dashboard"""
    
    # Hero metrics
    st.markdown("### 📊 Model Performance Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Universe Size",
            f"{len(predictions['ticker'].unique())} stocks",
            help="Total number of stocks analyzed"
        )
    
    with col2:
        st.metric(
            "Time Periods",
            f"{len(predictions['date'].unique())} months",
            help="Number of months in analysis"
        )
    
    with col3:
        if oot_metrics is not None and 'OOT_Mean_IC' in oot_metrics.columns:
            ic_value = oot_metrics['OOT_Mean_IC'].iloc[0]
            st.metric(
                "Mean IC",
                f"{ic_value:.4f}",
                delta=f"{(ic_value - 0.001)*100:.1f}% vs baseline",
                help="Information Coefficient - correlation with returns"
            )
        else:
            st.metric("Mean IC", "N/A")
    
    with col4:
        if oot_metrics is not None and 'OOT_Sharpe' in oot_metrics.columns:
            sharpe = oot_metrics['OOT_Sharpe'].iloc[0]
            st.metric(
                "Sharpe Ratio",
                f"{sharpe:.3f}",
                delta=f"+{(sharpe - 0.864)*100:.1f}% vs baseline",
                help="Risk-adjusted returns"
            )
        else:
            st.metric("Sharpe Ratio", "N/A")
    
    with col5:
        if turnover_data is not None and 'turnover' in turnover_data.columns:
            turnover = turnover_data['turnover'].mean() * 12 * 100
            delta_turnover = 419 - turnover
            st.metric(
                "Annual Turnover",
                f"{turnover:.0f}%",
                delta=f"-{delta_turnover:.0f}% vs baseline",
                delta_color="inverse",
                help="Portfolio turnover rate (lower is better)"
            )
        else:
            st.metric("Annual Turnover", "N/A")
    
    st.markdown("---")
    
    # Main visualizations
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📈 Prediction Distribution Over Time")
        
        # Calculate monthly statistics
        monthly_stats = predictions.groupby('date').agg({
            'prediction': ['mean', 'std', 'min', 'max', 'count']
        }).reset_index()
        monthly_stats.columns = ['date', 'mean', 'std', 'min', 'max', 'count']
        
        fig = go.Figure()
        
        # Add mean line
        fig.add_trace(go.Scatter(
            x=monthly_stats['date'],
            y=monthly_stats['mean'],
            mode='lines',
            name='Mean',
            line=dict(color='#1f77b4', width=3)
        ))
        
        # Add confidence band
        fig.add_trace(go.Scatter(
            x=monthly_stats['date'],
            y=monthly_stats['mean'] + monthly_stats['std'],
            mode='lines',
            name='+1 Std',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=monthly_stats['date'],
            y=monthly_stats['mean'] - monthly_stats['std'],
            mode='lines',
            name='-1 Std',
            line=dict(width=0),
            fillcolor='rgba(31, 119, 180, 0.2)',
            fill='tonexty',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Prediction",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Latest Month Snapshot")
        
        latest_date = predictions['date'].max()
        latest_preds = predictions[predictions['date'] == latest_date]
        
        # Key stats
        st.markdown(f"**Date:** {latest_date.strftime('%B %Y')}")
        st.markdown(f"**Stocks:** {len(latest_preds)}")
        st.markdown(f"**Mean Prediction:** {latest_preds['prediction'].mean():.4f}")
        st.markdown(f"**Std Prediction:** {latest_preds['prediction'].std():.4f}")
        
        st.markdown("---")
        
        # Histogram
        fig = px.histogram(
            latest_preds,
            x='prediction',
            nbins=30,
            title="Current Distribution",
            labels={'prediction': 'Predicted Return'}
        )
        fig.update_layout(showlegend=False, height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    # Bottom section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 15 Stocks (Latest Month)")
        top_stocks = latest_preds.nlargest(15, 'prediction')[['ticker', 'sector', 'prediction']]
        top_stocks['rank'] = range(1, len(top_stocks) + 1)
        top_stocks['prediction'] = top_stocks['prediction'].apply(lambda x: f"{x:.4f}")
        top_stocks = top_stocks[['rank', 'ticker', 'sector', 'prediction']]
        st.dataframe(top_stocks, hide_index=True, use_container_width=True, height=400)
    
    with col2:
        st.subheader("📊 Sector Distribution")
        
        sector_stats = latest_preds.groupby('sector').agg({
            'prediction': ['count', 'mean']
        }).reset_index()
        sector_stats.columns = ['Sector', 'Count', 'Avg Prediction']
        sector_stats = sector_stats.sort_values('Avg Prediction', ascending=False)
        
        fig = px.bar(
            sector_stats,
            x='Avg Prediction',
            y='Sector',
            orientation='h',
            title="Average Prediction by Sector",
            color='Avg Prediction',
            color_continuous_scale='RdYlGn',
            text='Count'
        )
        fig.update_traces(texttemplate='%{text} stocks', textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_stock_analysis(predictions, shap_values):
    """Stock-level analysis"""
    st.header("Stock Analysis")
    
    tickers = sorted(predictions['ticker'].unique())
    
    selected_tickers = st.multiselect(
        "Select Stock(s) to Analyze",
        options=tickers,
        default=tickers[:3] if len(tickers) >= 3 else tickers,
        help="Select one or more stocks to view predictions and explanations"
    )
    
    if not selected_tickers:
        st.warning("Please select at least one stock")
        return
    
    for ticker in selected_tickers:
        with st.expander(f"📊 {ticker}", expanded=len(selected_tickers) == 1):
            ticker_data = predictions[predictions['ticker'] == ticker].sort_values('date')
            
            if len(ticker_data) == 0:
                st.warning(f"No data for {ticker}")
                continue
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ticker_data['date'],
                    y=ticker_data['prediction'],
                    mode='lines+markers',
                    name='Prediction',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                fig.update_layout(
                    title=f"{ticker} - Prediction Time Series",
                    xaxis_title="Date",
                    yaxis_title="Predicted Return",
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Latest Prediction**")
                latest = ticker_data.iloc[-1]
                st.metric("Prediction", f"{latest['prediction']:.4f}")
                st.metric("Sector", latest['sector'])
                st.metric("Date", latest['date'].strftime('%Y-%m'))
                
                rank = (predictions[predictions['date'] == latest['date']]['prediction'] > latest['prediction']).sum() + 1
                total = len(predictions[predictions['date'] == latest['date']])
                st.metric("Rank", f"{rank}/{total}")
            
            if shap_values is not None:
                st.markdown("**Feature Contributions (SHAP Values)**")
                ticker_shap = shap_values[shap_values['ticker'] == ticker]
                
                if len(ticker_shap) > 0:
                    latest_shap = ticker_shap.iloc[-1]
                    feature_cols = [col for col in ticker_shap.columns if col not in ['ticker', 'date']]
                    
                    shap_data = pd.DataFrame({
                        'Feature': feature_cols,
                        'SHAP Value': [latest_shap[col] for col in feature_cols]
                    }).sort_values('SHAP Value', key=abs, ascending=False).head(10)
                    
                    fig = px.bar(
                        shap_data,
                        x='SHAP Value',
                        y='Feature',
                        orientation='h',
                        title=f"Top 10 Feature Contributions for {ticker}",
                        color='SHAP Value',
                        color_continuous_scale='RdBu_r'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No SHAP values available for this stock")

def show_explainability(shap_values, feature_importance):
    """Model explainability"""
    st.header("Model Explainability")
    
    if feature_importance is not None:
        st.subheader("Feature Importance (Information Coefficient)")
        
        fig = px.bar(
            feature_importance.sort_values('mean_ic', ascending=True).tail(15),
            x='mean_ic',
            y='feature',
            orientation='h',
            title="Top 15 Features by Mean IC",
            labels={'mean_ic': 'Mean IC', 'feature': 'Feature'},
            color='mean_ic',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Feature Categories**")
            feature_categories = {
                'Momentum': ['Mom_1', 'Mom_6_1', 'Mom_12_1'],
                'Volatility': ['Vol_12', 'IdioVol', 'Beta_12', 'MaxRet_1M', 'VolRatio'],
                'Value': ['PE_TTM', 'PB', 'EV_EBITDA'],
                'Quality': ['ROE', 'ROA', 'AssetTurnover'],
                'Growth': ['RevGrowth_YoY', 'EarnGrowth_YoY'],
                'Size': ['LogMktCap'],
                'Liquidity': ['Turnover_3M', 'Amihud']
            }
            
            category_ic = {}
            for category, features in feature_categories.items():
                cat_features = feature_importance[feature_importance['feature'].isin(features)]
                if len(cat_features) > 0:
                    category_ic[category] = cat_features['mean_ic'].mean()
            
            cat_df = pd.DataFrame(list(category_ic.items()), columns=['Category', 'Mean IC'])
            cat_df = cat_df.sort_values('Mean IC', ascending=False)
            
            fig = px.bar(
                cat_df,
                x='Mean IC',
                y='Category',
                orientation='h',
                title="Feature Category Performance"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Feature Statistics**")
            # Only show columns that exist
            available_cols = ['feature', 'mean_ic', 'ic_ir']
            if 'stability' in feature_importance.columns:
                available_cols.append('stability')
            st.dataframe(
                feature_importance[available_cols].sort_values('mean_ic', ascending=False),
                hide_index=True,
                use_container_width=True
            )
    
    if shap_values is not None:
        st.subheader("SHAP Value Analysis")
        
        feature_cols = [col for col in shap_values.columns if col not in ['ticker', 'date']]
        
        shap_summary = pd.DataFrame({
            'Feature': feature_cols,
            'Mean |SHAP|': [shap_values[col].abs().mean() for col in feature_cols]
        }).sort_values('Mean |SHAP|', ascending=False).head(15)
        
        fig = px.bar(
            shap_summary,
            x='Mean |SHAP|',
            y='Feature',
            orientation='h',
            title="Feature Importance by Mean Absolute SHAP Value"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_performance(predictions, oot_metrics, turnover_data):
    """Performance metrics"""
    st.header("Model Performance")
    
    if oot_metrics is not None:
        st.subheader("Out-of-Time Validation Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'OOT_Mean_IC' in oot_metrics.columns:
                st.metric("Mean IC", f"{oot_metrics['OOT_Mean_IC'].iloc[0]:.4f}")
            else:
                st.metric("Mean IC", "N/A")
        with col2:
            if 'OOT_IC_IR' in oot_metrics.columns:
                st.metric("IC-IR", f"{oot_metrics['OOT_IC_IR'].iloc[0]:.4f}")
            else:
                st.metric("IC-IR", "N/A")
        with col3:
            if 'OOT_Sharpe' in oot_metrics.columns:
                st.metric("Sharpe Ratio", f"{oot_metrics['OOT_Sharpe'].iloc[0]:.3f}")
            else:
                st.metric("Sharpe Ratio", "N/A")
        with col4:
            if 'OOT_WinRate' in oot_metrics.columns:
                st.metric("Win Rate", f"{oot_metrics['OOT_WinRate'].iloc[0]*100:.1f}%")
            else:
                st.metric("Win Rate", "N/A")
        
        st.dataframe(oot_metrics, hide_index=True, use_container_width=True)
    
    if turnover_data is not None:
        st.subheader("Turnover Analysis")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=turnover_data['date'] if 'date' in turnover_data.columns else turnover_data.index,
            y=turnover_data['turnover'] * 100,
            mode='lines+markers',
            name='Monthly Turnover',
            line=dict(color='#ff7f0e', width=2)
        ))
        
        fig.add_hline(
            y=25,
            line_dash="dash",
            line_color="red",
            annotation_text="Target (25%/month)"
        )
        
        fig.update_layout(
            title="Monthly Turnover Over Time",
            xaxis_title="Period",
            yaxis_title="Turnover (%)",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Monthly Turnover", f"{turnover_data['turnover'].mean()*100:.1f}%")
        with col2:
            st.metric("Annual Turnover", f"{turnover_data['turnover'].mean()*12*100:.1f}%")
    
    st.subheader("Prediction Statistics")
    
    monthly_stats = predictions.groupby('date').agg({
        'prediction': ['mean', 'std', 'min', 'max']
    }).reset_index()
    monthly_stats.columns = ['date', 'mean', 'std', 'min', 'max']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_stats['date'],
        y=monthly_stats['mean'],
        mode='lines',
        name='Mean',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=monthly_stats['date'],
        y=monthly_stats['mean'] + monthly_stats['std'],
        mode='lines',
        name='+1 Std',
        line=dict(color='lightblue', width=1, dash='dash'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=monthly_stats['date'],
        y=monthly_stats['mean'] - monthly_stats['std'],
        mode='lines',
        name='-1 Std',
        line=dict(color='lightblue', width=1, dash='dash'),
        fill='tonexty',
        showlegend=False
    ))
    
    fig.update_layout(
        title="Prediction Distribution Over Time",
        xaxis_title="Date",
        yaxis_title="Prediction",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

def show_configuration():
    """Configuration details"""
    st.header("Model Configuration")
    
    config_files = list(Path('configs').glob('*.json'))
    
    if config_files:
        selected_config = st.selectbox(
            "Select Configuration",
            options=[f.stem for f in config_files],
            index=0
        )
        
        config_path = Path('configs') / f"{selected_config}.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        st.subheader(f"Configuration: {selected_config}")
        st.markdown(f"**Description:** {config.get('description', 'N/A')}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model Settings**")
            st.json(config.get('models', {}))
            
            st.markdown("**Feature Settings**")
            st.json(config.get('features', {}))
        
        with col2:
            st.markdown("**Turnover Settings**")
            st.json(config.get('turnover', {}))
            
            st.markdown("**Portfolio Settings**")
            st.json(config.get('portfolio', {}))
        
        with st.expander("View Full Configuration"):
            st.json(config)
    else:
        st.warning("No configuration files found")
    
    st.markdown("---")
    st.subheader("System Information")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Python Version", "3.8+")
    with col2:
        st.metric("Model Type", "Ensemble (Ridge + LightGBM)")
    with col3:
        st.metric("Update Frequency", "Monthly")

if __name__ == "__main__":
    main()
