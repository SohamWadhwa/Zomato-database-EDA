import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import kagglehub

# Load and clean data
def load_data():
    # Download the dataset using kagglehub
    path = kagglehub.dataset_download("rajeshrampure/zomato-dataset")
    # Find the CSV file in the downloaded path
    import os
    csv_path = None
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(root, file)
                break
        if csv_path:
            break
    if not csv_path:
        raise FileNotFoundError("No CSV file found in the downloaded dataset.")
    df = pd.read_csv(csv_path)
    df['rate'] = pd.to_numeric(df['rate'].astype(str).str.extract(r'(\d+\.?\d*)', expand=False), errors='coerce')
    if 'approx_cost(for two people)' in df.columns:
        df['approx_cost(for two people)'] = pd.to_numeric(df['approx_cost(for two people)'].astype(str).str.replace(',', ''), errors='coerce')
    if 'votes' in df.columns:
        df['votes'] = pd.to_numeric(df['votes'], errors='coerce')
    df['city'] = df['listed_in(city)'] if 'listed_in(city)' in df.columns else np.nan
    subset_cols = [c for c in ['name', 'address', 'location'] if c in df.columns]
    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols, keep='first')
    return df

df = load_data()

st.title('Restaurant Ratings EDA Dashboard')
st.markdown('---')

# Sidebar filters
locations = df['location'].dropna().unique()
selected_location = st.sidebar.selectbox('Select Location', ['All'] + sorted(locations.tolist()))

cuisines = []
if 'cuisines' in df.columns:
    cuisines = pd.Series(df['cuisines'].dropna().str.split(',')).explode().str.strip().unique()
selected_cuisine = st.sidebar.selectbox('Select Cuisine', ['All'] + sorted([c for c in cuisines if c]))

# Filter data
filtered_df = df.copy()
if selected_location != 'All':
    filtered_df = filtered_df[filtered_df['location'] == selected_location]
if selected_cuisine != 'All' and 'cuisines' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['cuisines'].str.contains(selected_cuisine, na=False)]

# 1. Ratings Distribution
st.header('Ratings Distribution')
fig1 = px.histogram(filtered_df, x='rate', nbins=20, title='Distribution of Ratings', color_discrete_sequence=['#636EFA'])
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.box(filtered_df, y='rate', title='Ratings Boxplot', color_discrete_sequence=['#EF553B'])
st.plotly_chart(fig2, use_container_width=True)

# 2. Cost Distribution
if 'approx_cost(for two people)' in filtered_df.columns:
    st.header('Cost Distribution')
    fig3 = px.histogram(filtered_df, x='approx_cost(for two people)', nbins=30, title='Distribution of Approx. Cost (for two people)', color_discrete_sequence=['#00CC96'])
    st.plotly_chart(fig3, use_container_width=True)
    fig4 = px.box(filtered_df, y='approx_cost(for two people)', title='Cost Boxplot', color_discrete_sequence=['#AB63FA'])
    st.plotly_chart(fig4, use_container_width=True)

# 3. Votes Distribution
if 'votes' in filtered_df.columns:
    st.header('Votes Distribution')
    fig5 = px.histogram(filtered_df, x='votes', nbins=30, title='Distribution of Votes', color_discrete_sequence=['#FFA15A'])
    st.plotly_chart(fig5, use_container_width=True)

# 4. Top Locations
if 'location' in filtered_df.columns:
    st.header('Top Locations by Restaurant Count')
    top_locs = filtered_df['location'].value_counts().head(15)
    fig6 = px.bar(top_locs, x=top_locs.values, y=top_locs.index, orientation='h', title='Top 15 Locations')
    st.plotly_chart(fig6, use_container_width=True)

# 5. Top Cuisines
if 'cuisines' in filtered_df.columns:
    st.header('Top Cuisines')
    cuisines_expl = filtered_df['cuisines'].dropna().str.split(',').explode().str.strip()
    top_cuis = cuisines_expl.value_counts().head(15)
    fig7 = px.bar(top_cuis, x=top_cuis.values, y=top_cuis.index, orientation='h', title='Top 15 Cuisines')
    st.plotly_chart(fig7, use_container_width=True)

# 6. Price Bands
if 'approx_cost(for two people)' in filtered_df.columns:
    st.header('Price Band Analysis')
    filtered_df['price_band'] = pd.cut(filtered_df['approx_cost(for two people)'], bins=[-np.inf, 500, 1500, np.inf], labels=['Budget', 'Medium', 'High'])
    fig8 = px.histogram(filtered_df, x='price_band', color='price_band', title='Restaurant Count by Price Band')
    st.plotly_chart(fig8, use_container_width=True)
    fig9 = px.box(filtered_df, x='price_band', y='rate', color='price_band', title='Rating by Price Band')
    st.plotly_chart(fig9, use_container_width=True)

# 7. Ratings vs Cost
if {'rate', 'approx_cost(for two people)'}.issubset(filtered_df.columns):
    st.header('Ratings vs Cost')
    fig10 = px.scatter(filtered_df, x='approx_cost(for two people)', y='rate', trendline='ols', title='Rating vs Approx. Cost (for two people)', color='price_band')
    st.plotly_chart(fig10, use_container_width=True)

# 8. Ratings vs Votes
if {'rate', 'votes'}.issubset(filtered_df.columns):
    st.header('Ratings vs Votes')
    fig11 = px.scatter(filtered_df, x='votes', y='rate', trendline='ols', title='Rating vs Votes')
    st.plotly_chart(fig11, use_container_width=True)

# 9. Correlation Heatmap
num_cols = filtered_df.select_dtypes(include=[np.number]).columns
if len(num_cols) >= 2:
    st.header('Correlation Heatmap')
    corr = filtered_df[num_cols].corr(numeric_only=True)
    fig12 = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Blues'))
    fig12.update_layout(title='Correlation Heatmap (Numeric Features)')
    st.plotly_chart(fig12, use_container_width=True)

# 10. Top Restaurants Table
st.header('Highly Rated Popular Restaurants')
top_rest = filtered_df.dropna(subset=['rate', 'votes'])
top_rest = top_rest.query('rate >= 4.5').nlargest(10, 'votes')[['name', 'location', 'rate', 'votes']]
st.dataframe(top_rest)

# 11. Actionable Summary
st.header('Quick Insights')
if 'price_band' in filtered_df.columns:
    price_stats = filtered_df.groupby('price_band')['rate'].agg(['count', 'mean']).round(2)
    st.write('Average rating by price band:')
    st.dataframe(price_stats)

st.markdown('---')
st.markdown('**Tip:** Use the sidebar to filter by location and cuisine for more targeted insights.')
