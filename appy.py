import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("### ✅ App is booting successfully... Please upload your CSV file below or use the demo dataset.")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #ffffff;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1f2937;
        font-weight: 700;
    }
    h2, h3 {
        color: #374151;
    }
    </style>
""", unsafe_allow_html=True)

# === Step 4: Lazy import + safer parsing ===
@st.cache_data
def load_data(uploaded_file):
    import pandas as pd  # Lazy import for faster app startup

    df = pd.read_csv(uploaded_file)
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')

    if 'Order_Time' in df.columns:
        df['Order_Time'] = pd.to_datetime(df['Order_Time'], format='%H:%M:%S', errors='coerce').dt.time

    if 'Delivery_Date' in df.columns:
        df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'], errors='coerce')

    return df


# File upload section
st.sidebar.markdown("---")
st.sidebar.title("📤 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload your sales CSV file",
    type=['csv'],
    help="Upload the comprehensive_sales_data_sept_oct_2025.csv file"
)

# === Step 3: Demo dataset fallback ===
use_demo = st.sidebar.checkbox("🧪 Use Sample Dataset (for testing)", value=False)

if uploaded_file is not None:
    df = load_data(uploaded_file)
elif use_demo:
    import pandas as pd, numpy as np
    st.sidebar.info("🧪 Using sample demo dataset")
    dates = pd.date_range("2025-09-01", periods=30)
    df = pd.DataFrame({
        "Order_ID": range(1, 31),
        "Order_Date": dates,
        "Order_Time": ["12:00:00"] * 30,
        "Delivery_Date": dates + pd.Timedelta(days=2),
        "Product_Category": np.random.choice(["Hair", "Skin", "Health"], 30),
        "Customer_City": np.random.choice(["Mumbai", "Delhi", "Bangalore", "Pune"], 30),
        "Customer_Segment": np.random.choice(["Premium", "Regular"], 30),
        "Payment_Method": np.random.choice(["Credit Card", "UPI", "COD"], 30),
        "Total_Amount_INR": np.random.randint(1000, 5000, 30),
        "Order_Status": np.random.choice(["Delivered", "Processing", "In Transit"], 30)
    })
else:
    st.warning("⚠️ Please upload a CSV file or enable demo mode from the sidebar.")
    st.stop()


# Load the data
df = load_data(uploaded_file)

# Show success message
st.sidebar.success(f"✅ Dataset loaded successfully!")
st.sidebar.info(f"📊 Total Records: {len(df):,}")
st.sidebar.info(f"📅 Date Range: {df['Order_Date'].min().date()} to {df['Order_Date'].max().date()}")

# Sidebar filters
st.sidebar.image("https://via.placeholder.com/150x50/4CAF50/FFFFFF?text=Sales+Analytics", use_column_width=True)
st.sidebar.title("🔍 Filters")

# Date range filter
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(df['Order_Date'].min(), df['Order_Date'].max()),
    min_value=df['Order_Date'].min(),
    max_value=df['Order_Date'].max()
)

# Category filter
categories = ['All'] + sorted(df['Product_Category'].unique().tolist())
selected_category = st.sidebar.selectbox("Product Category", categories)

# City filter
cities = ['All'] + sorted(df['Customer_City'].unique().tolist())
selected_city = st.sidebar.multiselect("Customer City", cities, default=['All'])

# Customer segment filter
segments = ['All'] + df['Customer_Segment'].unique().tolist()
selected_segment = st.sidebar.selectbox("Customer Segment", segments)

# Payment method filter
payment_methods = ['All'] + df['Payment_Method'].unique().tolist()
selected_payment = st.sidebar.selectbox("Payment Method", payment_methods)

# Apply filters
filtered_df = df[
    (df['Order_Date'] >= pd.to_datetime(date_range[0])) &
    (df['Order_Date'] <= pd.to_datetime(date_range[1]))
]

if selected_category != 'All':
    filtered_df = filtered_df[filtered_df['Product_Category'] == selected_category]

if 'All' not in selected_city:
    filtered_df = filtered_df[filtered_df['Customer_City'].isin(selected_city)]

if selected_segment != 'All':
    filtered_df = filtered_df[filtered_df['Customer_Segment'] == selected_segment]

if selected_payment != 'All':
    filtered_df = filtered_df[filtered_df['Payment_Method'] == selected_payment]

# Main dashboard
st.title("📊 Sales Analytics Dashboard")
st.markdown("### Comprehensive Business Intelligence & Analytics")

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 Overview", 
    "📈 Sales Analysis", 
    "🛍️ Product Insights", 
    "👥 Customer Analytics",
    "🗺️ Geographic Analysis",
    "💰 Revenue Deep Dive",
    "⭐ Performance Metrics"
])

# ==================== TAB 1: OVERVIEW ====================
with tab1:
    st.header("Business Overview")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_revenue = filtered_df['Total_Amount_INR'].sum()
    total_orders = len(filtered_df)
    avg_order_value = filtered_df['Total_Amount_INR'].mean()
    total_customers = filtered_df['Customer_ID'].nunique()
    total_products_sold = filtered_df['Quantity'].sum()
    
    with col1:
        st.metric(
            label="💰 Total Revenue",
            value=f"₹{total_revenue:,.0f}",
            delta=f"{(total_revenue/1000000):.2f}M"
        )
    
    with col2:
        st.metric(
            label="🛒 Total Orders",
            value=f"{total_orders:,}",
            delta="Active"
        )
    
    with col3:
        st.metric(
            label="📊 Avg Order Value",
            value=f"₹{avg_order_value:,.0f}",
            delta=f"{((avg_order_value - df['Total_Amount_INR'].mean()) / df['Total_Amount_INR'].mean() * 100):.1f}%"
        )
    
    with col4:
        st.metric(
            label="👥 Total Customers",
            value=f"{total_customers:,}",
            delta="Unique"
        )
    
    with col5:
        st.metric(
            label="📦 Products Sold",
            value=f"{total_products_sold:,}",
            delta="Units"
        )
    
    st.markdown("---")
    
    # Row 2: Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily sales trend
        daily_sales = filtered_df.groupby('Order_Date').agg({
            'Total_Amount_INR': 'sum',
            'Order_ID': 'count'
        }).reset_index()
        daily_sales.columns = ['Date', 'Revenue', 'Orders']
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=daily_sales['Date'], y=daily_sales['Revenue'], 
                      name="Revenue", line=dict(color='#4CAF50', width=3),
                      fill='tozeroy', fillcolor='rgba(76, 175, 80, 0.1)'),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=daily_sales['Date'], y=daily_sales['Orders'], 
                      name="Orders", line=dict(color='#FF9800', width=2)),
            secondary_y=True
        )
        
        fig.update_layout(
            title="Daily Sales Trend",
            title_font=dict(size=18, color='#1f2937'),
            height=400,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(title_text="Date", showgrid=True, gridcolor='#f0f0f0')
        fig.update_yaxes(title_text="Revenue (₹)", secondary_y=False, showgrid=True, gridcolor='#f0f0f0')
        fig.update_yaxes(title_text="Number of Orders", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Order status distribution
        status_dist = filtered_df['Order_Status'].value_counts()
        
        colors = {'Delivered': '#4CAF50', 'In Transit': '#FF9800', 'Processing': '#2196F3'}
        
        fig = go.Figure(data=[go.Pie(
            labels=status_dist.index,
            values=status_dist.values,
            hole=0.4,
            marker=dict(colors=[colors.get(x, '#999') for x in status_dist.index]),
            textinfo='label+percent',
            textfont=dict(size=14),
            hovertemplate='%{label}<br>Orders: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Order Status Distribution",
            title_font=dict(size=18, color='#1f2937'),
            height=400,
            showlegend=True,
            paper_bgcolor='white',
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Row 3: More insights
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 products by revenue
        top_products = filtered_df.groupby('Product_Name')['Total_Amount_INR'].sum().sort_values(ascending=True).tail(10)
        
        fig = go.Figure(go.Bar(
            x=top_products.values,
            y=top_products.index,
            orientation='h',
            marker=dict(
                color=top_products.values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Revenue (₹)")
            ),
            text=[f"₹{x:,.0f}" for x in top_products.values],
            textposition='auto',
            hovertemplate='%{y}<br>Revenue: ₹%{x:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Top 10 Products by Revenue",
            title_font=dict(size=18, color='#1f2937'),
            height=450,
            xaxis_title="Revenue (₹)",
            yaxis_title="",
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False
        )
        fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sales by category
        category_sales = filtered_df.groupby('Product_Category')['Total_Amount_INR'].sum().sort_values(ascending=False).head(10)
        
        fig = go.Figure(data=[go.Bar(
            x=category_sales.index,
            y=category_sales.values,
            marker=dict(
                color=category_sales.values,
                colorscale='Blues',
                showscale=False
            ),
            text=[f"₹{x/1000:.0f}K" for x in category_sales.values],
            textposition='auto',
            hovertemplate='%{x}<br>Revenue: ₹%{y:,.0f}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Top 10 Categories by Revenue",
            title_font=dict(size=18, color='#1f2937'),
            height=450,
            xaxis_title="",
            yaxis_title="Revenue (₹)",
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        fig.update_xaxes(tickangle=-45, showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
        
        st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 2: SALES ANALYSIS ====================
with tab2:
    st.header("Sales Performance Analysis")
    
    # Sales metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_discount = filtered_df['Discount_Amount_INR'].sum()
        st.metric("💸 Total Discounts", f"₹{total_discount:,.0f}")
    
    with col2:
        total_tax = filtered_df['Tax_Amount_INR'].sum()
        st.metric("📋 Total Tax Collected", f"₹{total_tax:,.0f}")
    
    with col3:
        avg_discount = filtered_df['Discount_Percent'].mean()
        st.metric("🏷️ Avg Discount %", f"{avg_discount:.1f}%")
    
    with col4:
        conversion_rate = (filtered_df['Payment_Status'] == 'Completed').sum() / len(filtered_df) * 100
        st.metric("✅ Conversion Rate", f"{conversion_rate:.1f}%")
    
    st.markdown("---")
    
    # Sales channel performance
    col1, col2 = st.columns(2)
    
    with col1:
        channel_data = filtered_df.groupby('Sales_Channel').agg({
            'Total_Amount_INR': 'sum',
            'Order_ID': 'count'
        }).reset_index()
        channel_data.columns = ['Channel', 'Revenue', 'Orders']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Revenue by Channel', 'Orders by Channel'),
            specs=[[{'type':'domain'}, {'type':'domain'}]]
        )
        
        fig.add_trace(go.Pie(
            labels=channel_data['Channel'],
            values=channel_data['Revenue'],
            name="Revenue",
            marker=dict(colors=['#4CAF50', '#2196F3']),
            textinfo='label+percent'
        ), 1, 1)
        
        fig.add_trace(go.Pie(
            labels=channel_data['Channel'],
            values=channel_data['Orders'],
            name="Orders",
            marker=dict(colors=['#FF9800', '#9C27B0']),
            textinfo='label+percent'
        ), 1, 2)
        
        fig.update_layout(
            title="Sales Channel Performance",
            title_font=dict(size=18, color='#1f2937'),
            height=400,
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Payment method analysis
        payment_data = filtered_df.groupby('Payment_Method').agg({
            'Total_Amount_INR': 'sum',
            'Order_ID': 'count'
        }).reset_index()
        payment_data.columns = ['Payment Method', 'Revenue', 'Orders']
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Revenue',
            x=payment_data['Payment Method'],
            y=payment_data['Revenue'],
            marker_color='#4CAF50',
            text=[f"₹{x/1000:.0f}K" for x in payment_data['Revenue']],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='Orders',
            x=payment_data['Payment Method'],
            y=payment_data['Orders'],
            marker_color='#2196F3',
            text=payment_data['Orders'],
            textposition='auto',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Payment Method Analysis",
            title_font=dict(size=18, color='#1f2937'),
            height=400,
            xaxis_title="Payment Method",
            yaxis=dict(title="Revenue (₹)", side='left'),
            yaxis2=dict(title="Orders", side='right', overlaying='y'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Hourly sales pattern
    st.subheader("Sales Pattern by Hour")
    
    filtered_df['Hour'] = pd.to_datetime(filtered_df['Order_Time'], format='%H:%M:%S').dt.hour
    hourly_sales = filtered_df.groupby('Hour').agg({
        'Total_Amount_INR': 'sum',
        'Order_ID': 'count'
    }).reset_index()
    hourly_sales.columns = ['Hour', 'Revenue', 'Orders']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hourly_sales['Hour'],
        y=hourly_sales['Revenue'],
        mode='lines+markers',
        name='Revenue',
        line=dict(color='#4CAF50', width=3),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor='rgba(76, 175, 80, 0.1)'
    ))
    
    fig.add_trace(go.Scatter(
        x=hourly_sales['Hour'],
        y=hourly_sales['Orders'],
        mode='lines+markers',
        name='Orders',
        line=dict(color='#FF9800', width=3),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Hourly Sales Pattern",
        title_font=dict(size=18, color='#1f2937'),
        height=400,
        xaxis=dict(title="Hour of Day", tickmode='linear', tick0=8, dtick=1),
        yaxis=dict(title="Revenue (₹)", side='left'),
        yaxis2=dict(title="Orders", side='right', overlaying='y'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Discount impact analysis
    col1, col2 = st.columns(2)
    
    with col1:
        discount_bins = pd.cut(filtered_df['Discount_Percent'], bins=[0, 5, 10, 15, 20], labels=['0-5%', '5-10%', '10-15%', '15-20%'])
        discount_impact = filtered_df.groupby(discount_bins).agg({
            'Order_ID': 'count',
            'Total_Amount_INR': 'mean'
        }).reset_index()
        discount_impact.columns = ['Discount Range', 'Orders', 'Avg Order Value']
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=discount_impact['Discount Range'],
            y=discount_impact['Orders'],
            name='Number of Orders',
            marker_color='#2196F3',
            text=discount_impact['Orders'],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Orders by Discount Range",
            title_font=dict(size=18, color='#1f2937'),
            height=400,
            xaxis_title="Discount Range",
            yaxis_title="Number of Orders",
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Shipping method preference
        shipping_data = filtered_df.groupby('Shipping_Method').agg({
            'Order_ID': 'count',
            'Total_Amount_INR': 'sum'
        }).reset_index()
        shipping_data.columns = ['Shipping Method', 'Orders', 'Revenue']
        
        fig = go.Figure(data=[go.Pie(
            labels=shipping_data['Shipping Method'],
            values=shipping_data['Orders'],
            hole=0.4,
            marker=dict(colors=['#4CAF50', '#FF9800']),
            textinfo='label+percent+value',
            textfont=dict(size=14)
        )])
        
        fig.update_layout(
            title="Shipping Method Distribution",
            title_font=dict(size=18, color='#1f2937'),
            height=400,
            paper_bgcolor='white',
            annotations=[dict(text='Orders', x=0.5, y=0.5, font_size=16, showarrow=False)]
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 3: PRODUCT INSIGHTS ====================
with tab3:
    st.header("Product Performance Insights")
    
    # Product metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_products = filtered_df['Product_Name'].nunique()
        st.metric("🏷️ Unique Products", f"{total_products}")
    
    with col2:
        total_categories = filtered_df['Product_Category'].nunique()
        st.metric("📂 Product Categories", f"{total_categories}")
    
    with col3:
        avg_price = filtered_df['Unit_Price_INR'].mean()
        st.metric("💵 Avg Product Price", f"₹{avg_price:,.0f}")
    
    with col4:
        best_seller = filtered_df.groupby('Product_Name')['Quantity'].sum().idxmax()
        st.metric("🏆 Best Seller", "View Below")
    
    st.markdown("---")
    
    # Product performance table
    st.subheader("Top Performing Products")
    
    product_performance = filtered_df.groupby('Product_Name').agg({
        'Order_ID': 'count',
        'Quantity': 'sum',
        'Total_Amount_INR': 'sum',
        'Unit_Price_INR': 'mean',
        'Customer_Rating': lambda x: x[x > 0].mean() if (x > 0).any() else 0
    }).reset_index()
    product_performance.columns = ['Product', 'Orders', 'Units Sold', 'Revenue', 'Avg Price', 'Avg Rating']
    product_performance = product_performance.sort_values('Revenue', ascending=False).head(15)
    product_performance['Revenue'] = product_performance['Revenue'].apply(lambda x: f"₹{x:,.0f}")
    product_performance['Avg Price'] = product_performance['Avg Price'].apply(lambda x: f"₹{x:,.0f}")
    product_performance['Avg Rating'] = product_performance['Avg Rating'].apply(lambda x: f"{x:.1f}⭐" if x > 0 else "N/A")
    
    st.dataframe(product_performance, use_container_width=True, height=400)
    
    # Category analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Category revenue treemap
        category_data = filtered_df.groupby('Product_Category').agg({
            'Total_Amount_INR': 'sum',
            'Order_ID': 'count'
        }).reset_index()
        category_data.columns = ['Category', 'Revenue', 'Orders']
        category_data = category_data.sort_values('Revenue', ascending=False).head(15)
        
        fig = px.treemap(
            category_data,
            path=['Category'],
            values='Revenue',
            color='Revenue',
            color_continuous_scale='Viridis',
            title='Revenue Distribution by Category'
        )
        
        fig.update_layout(
            title_font=dict(size=18, color='#1f2937'),
            height=500,
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Products by quantity sold
        top_quantity = filtered_df.groupby('Product_Name')['Quantity'].sum().sort_values(ascending=False).head(10)
        
        fig = go.Figure(go.Bar(
            y=top_quantity.index,
            x=top_quantity.values,
            orientation='h',
            marker=dict(
                color=top_quantity.values,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Units")
            ),
            text=top_quantity.values,
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Top 10 Products by Units Sold",
            title_font=dict(size=18, color='#1f2937'),
            height=500,
            xaxis_title="Units Sold",
            yaxis_title="",
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Price vs Sales analysis
    st.subheader("Price Point Analysis")
    
    price_analysis = filtered_df.groupby('Unit_Price_INR').agg({
        'Order_ID': 'count',
        'Total_Amount_INR': 'sum'
    }).reset_index()
    price_analysis.columns = ['Price', 'Orders', 'Revenue']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=price_analysis['Price'],
        y=price_analysis['Orders'],
        mode='markers',
        marker=dict(
            size=price_analysis['Revenue']/10000,
            color=price_analysis['Revenue'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Revenue (₹)"),
            line=dict(width=1, color='white')
        ),
        text=[f"Price: ₹{p}<br>Orders: {o}<br>Revenue: ₹{r:,.0f}" 
              for p, o, r in zip(price_analysis['Price'], price_analysis['Orders'], price_analysis['Revenue'])],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Price Point vs Order Volume (Bubble size = Revenue)",
        title_font=dict(size=18, color='#1f2937'),
        height=450,
        xaxis_title="Product Price (₹)",
        yaxis_title="Number of Orders",
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Category comparison
    st.subheader("Category Performance Comparison")
    
    category_comparison = filtered_df.groupby('Product_Category').agg({
        'Total_Amount_INR': 'sum',
        'Order_ID': 'count',
        'Quantity': 'sum',
        'Unit_Price_INR': 'mean'
    }).reset_index()
    category_comparison.columns = ['Category', 'Revenue', 'Orders', 'Units', 'Avg Price']
    category_comparison = category_comparison.sort_values('Revenue', ascending=False).head(12)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Revenue by Category', 'Orders by Category', 
                       'Units Sold by Category', 'Average Price by Category'),
        specs=[[{'type':'bar'}, {'type':'bar'}],
               [{'type':'bar'}, {'type':'bar'}]]
    )
    
    # Revenue
    fig.add_trace(go.Bar(
        x=category_comparison['Category'],
        y=category_comparison['Revenue'],
        marker_color='#4CAF50',
        name='Revenue',
        showlegend=False
    ), row=1, col=1)
    
    # Orders
    fig.add_trace(go.Bar(
        x=category_comparison['Category'],
        y=category_comparison['Orders'],
        marker_color='#2196F3',
        name='Orders',
        showlegend=False
    ), row=1, col=2)
    
    # Units
    fig.add_trace(go.Bar(
        x=category_comparison['Category'],
        y=category_comparison['Units'],
        marker_color='#FF9800',
        name='Units',
        showlegend=False
    ), row=2, col=1)
    
    # Avg Price
    fig.add_trace(go.Bar(
        x=category_comparison['Category'],
        y=category_comparison['Avg Price'],
        marker_color='#9C27B0',
        name='Avg Price',
        showlegend=False
    ), row=2, col=2)
    
    fig.update_layout(
        height=700,
        title_text="Category Performance Metrics",
        title_font=dict(size=18, color='#1f2937'),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.update_xaxes(tickangle=-45, showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    
    st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 4: CUSTOMER ANALYTICS ====================
with tab4:
    st.header("Customer Behavior & Demographics")
    
    # Customer metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        new_customers = filtered_df[filtered_df['First_Time_Customer'] == 'Yes']['Customer_ID'].nunique()
        st.metric("🆕 New Customers", f"{new_customers}")
    
    with col2:
        repeat_customers = filtered_df[filtered_df['First_Time_Customer'] == 'No']['Customer_ID'].nunique()
        st.metric("🔄 Repeat Customers", f"{repeat_customers}")
    
    with col3:
        avg_lifetime_orders = filtered_df['Customer_Lifetime_Orders'].mean()
        st.metric("📊 Avg Lifetime Orders", f"{avg_lifetime_orders:.1f}")
    
    with col4:
        avg_rating = filtered_df[filtered_df['Customer_Rating'] > 0]['Customer_Rating'].mean()
        st.metric("⭐ Avg Customer Rating", f"{avg_rating:.2f}")
    
    st.markdown("---")
    
    # Customer demographics
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        age_bins = pd.cut(filtered_df['Customer_Age'], bins=[20, 25, 30, 35, 40, 45, 50], 
                         labels=['20-25', '25-30', '30-35', '35-40', '40-45', '45-50'])
        age_dist = age_bins.value_counts().sort_index()
        
        fig = go.Figure(data=[go.Bar(
            x=age_dist.index,
            y=age_dist.values,
            marker=dict(
                color=age_dist.values,
                colorscale='Blues',
                showscale=False
            ),
            text=age_dist.values,
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Customer Age Distribution",
            title_font=dict(size=18, color='#1f2937'),
            height=400,
            xaxis_title="Age Group",
            yaxis_title="Number of Customers",
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gender distribution
        gender_data = filtered_df.groupby('Customer_Gender').agg({
            'Order_ID': 'count',
            'Total_Amount_INR': 'sum'
        }).reset_index()
        gender_data.columns = ['Gender', 'Orders', 'Revenue']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Orders by Gender', 'Revenue by Gender'),
            specs=[[{'type':'domain'}, {'type':'domain'}]]
        )
        
        fig.add_trace(go.Pie(
            labels=gender_data['Gender'],
            values=gender_data['Orders'],
            marker=dict(colors=['#2196F3', '#E91E63']),
            textinfo='label+percent',
            name="Orders"
        ), 1, 1)
        
        fig.add_trace(go.Pie(
            labels=gender_data['Gender'],
            values=gender_data['Revenue'],
            marker=dict(colors=['#4CAF50', '#FF9800']),
            textinfo='label+percent',
            name="Revenue"
        ), 1, 2)
        
        fig.update_layout(
            title="Gender-wise Analysis",
            title_font=dict(size=18, color='#1f2937'),
            height=400,
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Customer segment analysis
    col1, col2 = st.columns(2)
    
    with col1:
        segment_data = filtered_df.groupby('Customer_Segment').agg({
            'Order_ID': 'count',
            'Total_Amount_INR': ['sum', 'mean']
        }).reset_index()
        segment_data.columns = ['Segment', 'Orders', 'Total Revenue', 'Avg Order Value']
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Total Revenue',
            x=segment_data['Segment'],
            y=segment_data['Total Revenue'],
            marker_color='#4CAF50',
            text=[f"₹{x/1000:.0f}K" for x in segment_data['Total Revenue']],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='Avg Order Value',
            x=segment_data['Segment'],
            y=segment_data['Avg Order Value'],
            marker_color='#FF9800',
            text=[f"₹{x:,.0f}" for x in segment_data['Avg Order Value']],
            textposition='auto',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Customer Segment Performance",
            title_font=dict(size=18, color='#1f2937'),
            height=400,
            xaxis_title="Customer Segment",
            yaxis=dict(title="Total Revenue (₹)", side='left'),
            yaxis2=dict(title="Avg Order Value (₹)", side='right', overlaying='y'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Customer retention
        retention_data = filtered_df.groupby('First_Time_Customer').agg({
            'Order_ID': 'count',
            'Total_Amount_INR': 'sum'
        }).reset_index()
        retention_data.columns = ['Customer Type', 'Orders', 'Revenue']
        retention_data['Customer Type'] = retention_data['Customer Type'].map({'Yes': 'New', 'No': 'Repeat'})
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Orders',
            x=retention_data['Customer Type'],
            y=retention_data['Orders'],
            marker_color='#2196F3',
            text=retention_data['Orders'],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='Revenue',
            x=retention_data['Customer Type'],
            y=retention_data['Revenue'],
            marker_color='#4CAF50',
            text=[f"₹{x/1000:.0f}K" for x in retention_data['Revenue']],
            textposition='auto',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="New vs Repeat Customer Analysis",
            title_font=dict(size=18, color='#1f2937'),
            height=400,
            xaxis_title="Customer Type",
            yaxis=dict(title="Number of Orders", side='left'),
            yaxis2=dict(title="Revenue (₹)", side='right', overlaying='y'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Customer lifetime value analysis
    st.subheader("Customer Lifetime Value Analysis")
    
    clv_bins = pd.cut(filtered_df['Customer_Lifetime_Orders'], 
                     bins=[0, 2, 5, 10, 20], 
                     labels=['1-2 orders', '3-5 orders', '6-10 orders', '11+ orders'])
    clv_data = filtered_df.groupby(clv_bins).agg({
        'Customer_ID': 'nunique',
        'Total_Amount_INR': 'sum'
    }).reset_index()
    clv_data.columns = ['Order Frequency', 'Customers', 'Total Revenue']
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=clv_data['Order Frequency'], y=clv_data['Customers'], 
               name="Customers", marker_color='#2196F3',
               text=clv_data['Customers'], textposition='auto'),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=clv_data['Order Frequency'], y=clv_data['Total Revenue'], 
                  name="Revenue", line=dict(color='#4CAF50', width=3),
                  mode='lines+markers', marker=dict(size=10)),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Customer Lifetime Value Distribution",
        title_font=dict(size=18, color='#1f2937'),
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Order Frequency", showgrid=False)
    fig.update_yaxes(title_text="Number of Customers", secondary_y=False, showgrid=True, gridcolor='#f0f0f0')
    fig.update_yaxes(title_text="Total Revenue (₹)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Rating distribution
    st.subheader("Customer Satisfaction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        rating_dist = filtered_df[filtered_df['Customer_Rating'] > 0]['Customer_Rating'].value_counts().sort_index()
        
        fig = go.Figure(data=[go.Bar(
            x=[f"{int(x)} ⭐" for x in rating_dist.index],
            y=rating_dist.values,
            marker=dict(
                color=['#f44336', '#FF9800', '#FFC107', '#8BC34A', '#4CAF50'],
                line=dict(color='white', width=2)
            ),
            text=rating_dist.values,
            textposition='auto',
            hovertemplate='Rating: %{x}<br>Count: %{y}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Rating Distribution",
            title_font=dict(size=18, color='#1f2937'),
            height=400,
            xaxis_title="Rating",
            yaxis_title="Number of Reviews",
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top reviewers
        if len(filtered_df[filtered_df['Customer_Rating'] > 0]) > 0:
            top_reviewers = filtered_df[filtered_df['Customer_Rating'] > 0].groupby('Customer_Name').agg({
                'Customer_Rating': 'mean',
                'Order_ID': 'count'
            }).reset_index()
            top_reviewers.columns = ['Customer', 'Avg Rating', 'Reviews']
            top_reviewers = top_reviewers.sort_values('Reviews', ascending=False).head(10)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=top_reviewers['Reviews'],
                y=top_reviewers['Avg Rating'],
                mode='markers+text',
                marker=dict(
                    size=top_reviewers['Reviews']*5,
                    color=top_reviewers['Avg Rating'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Rating"),
                    line=dict(width=2, color='white')
                ),
                text=top_reviewers['Customer'],
                textposition='top center',
                hovertemplate='Customer: %{text}<br>Reviews: %{x}<br>Avg Rating: %{y:.1f}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Top Reviewers (Bubble size = Review count)",
                title_font=dict(size=18, color='#1f2937'),
                height=400,
                xaxis_title="Number of Reviews",
                yaxis_title="Average Rating",
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
            fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0', range=[0, 5.5])
            
            st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 5: GEOGRAPHIC ANALYSIS ====================
with tab5:
    st.header("Geographic Distribution & Insights")
    
    # Geographic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_cities = filtered_df['Customer_City'].nunique()
        st.metric("🏙️ Cities Covered", f"{total_cities}")
    
    with col2:
        total_states = filtered_df['Customer_State'].nunique()
        st.metric("🗺️ States Covered", f"{total_states}")
    
    with col3:
        top_city = filtered_df.groupby('Customer_City')['Total_Amount_INR'].sum().idxmax()
        st.metric("🏆 Top City", top_city)
    
    with col4:
        top_state = filtered_df.groupby('Customer_State')['Total_Amount_INR'].sum().idxmax()
        st.metric("🏆 Top State", top_state)
    
    st.markdown("---")
    
    # City-wise analysis
    col1, col2 = st.columns(2)
    
    with col1:
        city_data = filtered_df.groupby('Customer_City').agg({
            'Total_Amount_INR': 'sum',
            'Order_ID': 'count'
        }).reset_index()
        city_data.columns = ['City', 'Revenue', 'Orders']
        city_data = city_data.sort_values('Revenue', ascending=False).head(15)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=city_data['City'],
            y=city_data['Revenue'],
            marker=dict(
                color=city_data['Revenue'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Revenue (₹)")
            ),
            text=[f"₹{x/1000:.0f}K" for x in city_data['Revenue']],
            textposition='auto',
            hovertemplate='%{x}<br>Revenue: ₹%{y:,.0f}<br>Orders: ' + 
                         city_data['Orders'].astype(str) + '<extra></extra>'
        ))
        
        fig.update_layout(
            title="Top 15 Cities by Revenue",
            title_font=dict(size=18, color='#1f2937'),
            height=450,
            xaxis_title="City",
            yaxis_title="Revenue (₹)",
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        fig.update_xaxes(tickangle=-45, showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        state_data = filtered_df.groupby('Customer_State').agg({
            'Total_Amount_INR': 'sum',
            'Order_ID': 'count'
        }).reset_index()
        state_data.columns = ['State', 'Revenue', 'Orders']
        state_data = state_data.sort_values('Revenue', ascending=False).head(15)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=state_data['State'],
            x=state_data['Revenue'],
            orientation='h',
            marker=dict(
                color=state_data['Revenue'],
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Revenue (₹)")
            ),
            text=[f"₹{x/1000:.0f}K" for x in state_data['Revenue']],
            textposition='auto',
            hovertemplate='%{y}<br>Revenue: ₹%{x:,.0f}<br>Orders: ' + 
                         state_data['Orders'].astype(str) + '<extra></extra>'
        ))
        
        fig.update_layout(
            title="Top 15 States by Revenue",
            title_font=dict(size=18, color='#1f2937'),
            height=450,
            xaxis_title="Revenue (₹)",
            yaxis_title="",
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Geographic heatmap
    st.subheader("Geographic Revenue Heatmap")
    
    geo_matrix = filtered_df.pivot_table(
        values='Total_Amount_INR',
        index='Customer_State',
        columns='Customer_Segment',
        aggfunc='sum',
        fill_value=0
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=geo_matrix.values,
        x=geo_matrix.columns,
        y=geo_matrix.index,
        colorscale='Viridis',
        text=[[f"₹{val/1000:.0f}K" for val in row] for row in geo_matrix.values],
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Revenue (₹)")
    ))
    
    fig.update_layout(
        title="Revenue by State & Customer Segment",
        title_font=dict(size=18, color='#1f2937'),
        height=600,
        xaxis_title="Customer Segment",
        yaxis_title="State",
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # City performance metrics
    st.subheader("City Performance Metrics")
    
    city_metrics = filtered_df.groupby('Customer_City').agg({
        'Order_ID': 'count',
        'Total_Amount_INR': 'sum',
        'Customer_ID': 'nunique',
        'Customer_Rating': lambda x: x[x > 0].mean() if (x > 0).any() else 0
    }).reset_index()
    city_metrics.columns = ['City', 'Orders', 'Revenue', 'Customers', 'Avg Rating']
    city_metrics['AOV'] = city_metrics['Revenue'] / city_metrics['Orders']
    city_metrics = city_metrics.sort_values('Revenue', ascending=False).head(20)
    
    city_metrics['Revenue'] = city_metrics['Revenue'].apply(lambda x: f"₹{x:,.0f}")
    city_metrics['AOV'] = city_metrics['AOV'].apply(lambda x: f"₹{x:,.0f}")
    city_metrics['Avg Rating'] = city_metrics['Avg Rating'].apply(lambda x: f"{x:.1f}⭐" if x > 0 else "N/A")
    
    st.dataframe(city_metrics, use_container_width=True, height=500)
    
    # Shipping analysis by location
    st.subheader("Shipping Preferences by Region")
    
    shipping_by_state = filtered_df.groupby(['Customer_State', 'Shipping_Method']).size().reset_index(name='Count')
    shipping_by_state = shipping_by_state.sort_values('Count', ascending=False)
    
    fig = px.bar(
        shipping_by_state.head(30),
        x='Customer_State',
        y='Count',
        color='Shipping_Method',
        title='Shipping Method Preferences by State',
        color_discrete_map={'Express': '#4CAF50', 'Standard': '#2196F3'},
        barmode='group'
    )
    
    fig.update_layout(
        title_font=dict(size=18, color='#1f2937'),
        height=450,
        xaxis_title="State",
        yaxis_title="Number of Orders",
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(tickangle=-45, showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    
    st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 6: REVENUE DEEP DIVE ====================
with tab6:
    st.header("Revenue Analysis & Financial Insights")
    
    # Revenue metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        gross_revenue = filtered_df['Unit_Price_INR'].sum() * filtered_df['Quantity'].sum()
        st.metric("💵 Gross Revenue", f"₹{gross_revenue:,.0f}")
    
    with col2:
        net_revenue = filtered_df['Total_Amount_INR'].sum() - filtered_df['Shipping_Cost_INR'].sum()
        st.metric("💰 Net Revenue", f"₹{net_revenue:,.0f}")
    
    with col3:
        total_discount = filtered_df['Discount_Amount_INR'].sum()
        discount_rate = (total_discount / gross_revenue * 100) if gross_revenue > 0 else 0
        st.metric("🏷️ Total Discounts", f"₹{total_discount:,.0f}", delta=f"-{discount_rate:.1f}%")
    
    with col4:
        shipping_revenue = filtered_df['Shipping_Cost_INR'].sum()
        st.metric("🚚 Shipping Revenue", f"₹{shipping_revenue:,.0f}")
    
    st.markdown("---")
    
    # Revenue breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue components
        revenue_components = pd.DataFrame({
            'Component': ['Product Revenue', 'Tax Revenue', 'Shipping Revenue', 'Discounts'],
            'Amount': [
                (filtered_df['Total_Amount_INR'].sum() - filtered_df['Tax_Amount_INR'].sum() - filtered_df['Shipping_Cost_INR'].sum()),
                filtered_df['Tax_Amount_INR'].sum(),
                filtered_df['Shipping_Cost_INR'].sum(),
                -filtered_df['Discount_Amount_INR'].sum()
            ]
        })
        
        colors = ['#4CAF50', '#2196F3', '#FF9800', '#f44336']
        
        fig = go.Figure(data=[go.Bar(
            x=revenue_components['Component'],
            y=revenue_components['Amount'],
            marker=dict(color=colors),
            text=[f"₹{x/1000:.0f}K" if x > 0 else f"-₹{abs(x)/1000:.0f}K" for x in revenue_components['Amount']],
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Revenue Components Breakdown",
            title_font=dict(size=18, color='#1f2937'),
            height=400,
            xaxis_title="Component",
            yaxis_title="Amount (₹)",
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Revenue by payment status
        payment_status_revenue = filtered_df.groupby('Payment_Status')['Total_Amount_INR'].sum()
        
        fig = go.Figure(data=[go.Pie(
            labels=payment_status_revenue.index,
            values=payment_status_revenue.values,
            hole=0.4,
            marker=dict(colors=['#4CAF50', '#FF9800']),
            textinfo='label+value+percent',
            texttemplate='%{label}<br>₹%{value:,.0f}<br>%{percent}',
            textfont=dict(size=12)
        )])
        
        fig.update_layout(
            title="Revenue by Payment Status",
            title_font=dict(size=18, color='#1f2937'),
            height=400,
            paper_bgcolor='white',
            annotations=[dict(text=f'Total<br>₹{payment_status_revenue.sum()/1000:.0f}K', 
                            x=0.5, y=0.5, font_size=14, showarrow=False)]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Weekly revenue analysis
    st.subheader("Weekly Revenue Analysis")
    
    filtered_df['Week'] = filtered_df['Order_Date'].dt.isocalendar().week
    weekly_revenue = filtered_df.groupby('Week').agg({
        'Total_Amount_INR': 'sum',
        'Order_ID': 'count',
        'Discount_Amount_INR': 'sum'
    }).reset_index()
    weekly_revenue.columns = ['Week', 'Revenue', 'Orders', 'Discounts']
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=weekly_revenue['Week'], y=weekly_revenue['Revenue'], 
               name="Revenue", marker_color='#4CAF50',
               text=[f"₹{x/1000:.0f}K" for x in weekly_revenue['Revenue']],
               textposition='auto'),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=weekly_revenue['Week'], y=weekly_revenue['Orders'], 
                  name="Orders", line=dict(color='#FF9800', width=3),
                  mode='lines+markers', marker=dict(size=8)),
        secondary_y=True
    )
    
    fig.add_trace(
        go.Scatter(x=weekly_revenue['Week'], y=weekly_revenue['Discounts'], 
                  name="Discounts", line=dict(color='#f44336', width=2, dash='dash'),
                  mode='lines+markers', marker=dict(size=6)),
        secondary_y=False
    )
    
    fig.update_layout(
        title="Weekly Revenue, Orders & Discounts Trend",
        title_font=dict(size=18, color='#1f2937'),
        height=450,
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Week Number", showgrid=True, gridcolor='#f0f0f0')
    fig.update_yaxes(title_text="Revenue / Discounts (₹)", secondary_y=False, showgrid=True, gridcolor='#f0f0f0')
    fig.update_yaxes(title_text="Number of Orders", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Revenue concentration
    col1, col2 = st.columns(2)
    
    with col1:
        # Top revenue generating products
        top_revenue_products = filtered_df.groupby('Product_Name')['Total_Amount_INR'].sum().sort_values(ascending=False).head(10)
        
        fig = go.Figure(data=[go.Pie(
            labels=top_revenue_products.index,
            values=top_revenue_products.values,
            textinfo='percent',
            hovertemplate='%{label}<br>Revenue: ₹%{value:,.0f}<br>Share: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Top 10 Products Revenue Share",
            title_font=dict(size=18, color='#1f2937'),
            height=450,
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Revenue by customer segment and payment method
        segment_payment = filtered_df.groupby(['Customer_Segment', 'Payment_Method'])['Total_Amount_INR'].sum().reset_index()
        
        fig = px.sunburst(
            segment_payment,
            path=['Customer_Segment', 'Payment_Method'],
            values='Total_Amount_INR',
            color='Total_Amount_INR',
            color_continuous_scale='Viridis',
            title='Revenue by Segment & Payment Method'
        )
        
        fig.update_layout(
            title_font=dict(size=18, color='#1f2937'),
            height=450,
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Revenue forecasting (simple moving average)
    st.subheader("Revenue Trend & Simple Forecast")
    
    daily_revenue = filtered_df.groupby('Order_Date')['Total_Amount_INR'].sum().reset_index()
    daily_revenue.columns = ['Date', 'Revenue']
    daily_revenue['MA_7'] = daily_revenue['Revenue'].rolling(window=7).mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_revenue['Date'],
        y=daily_revenue['Revenue'],
        mode='lines',
        name='Daily Revenue',
        line=dict(color='lightgray', width=1),
        opacity=0.5
    ))
    
    fig.add_trace(go.Scatter(
        x=daily_revenue['Date'],
        y=daily_revenue['MA_7'],
        mode='lines',
        name='7-Day Moving Average',
        line=dict(color='#4CAF50', width=3)
    ))
    
    fig.update_layout(
        title="Daily Revenue with 7-Day Moving Average",
        title_font=dict(size=18, color='#1f2937'),
        height=400,
        xaxis_title="Date",
        yaxis_title="Revenue (₹)",
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    
    st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 7: PERFORMANCE METRICS ====================
with tab7:
    st.header("Key Performance Indicators & Metrics")
    
    # KPI Overview
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        order_fulfillment = (filtered_df['Order_Status'] == 'Delivered').sum() / len(filtered_df) * 100
        st.metric("📦 Order Fulfillment", f"{order_fulfillment:.1f}%", delta="Target: 95%")
    
    with col2:
        delivered_df = filtered_df[filtered_df['Days_to_Deliver'] != '']
        if len(delivered_df) > 0:
            avg_delivery = delivered_df['Days_to_Deliver'].astype(float).mean()
            st.metric("🚚 Avg Delivery Days", f"{avg_delivery:.1f}", delta="Target: ≤5")
        else:
            st.metric("🚚 Avg Delivery Days", "N/A")
    
    with col3:
        customer_satisfaction = (filtered_df['Customer_Rating'] >= 4).sum() / len(filtered_df[filtered_df['Customer_Rating'] > 0]) * 100 if len(filtered_df[filtered_df['Customer_Rating'] > 0]) > 0 else 0
        st.metric("😊 Customer Satisfaction", f"{customer_satisfaction:.1f}%", delta="4+ stars")
    
    with col4:
        repeat_rate = (filtered_df['First_Time_Customer'] == 'No').sum() / len(filtered_df) * 100
        st.metric("🔄 Repeat Purchase Rate", f"{repeat_rate:.1f}%")
    
    with col5:
        premium_rate = (filtered_df['Customer_Segment'] == 'Premium').sum() / len(filtered_df) * 100
        st.metric("💎 Premium Customer %", f"{premium_rate:.1f}%")
    
    st.markdown("---")
    
    # Marketing campaign performance
    st.subheader("Marketing Campaign Performance")
    
    campaign_performance = filtered_df.groupby('Marketing_Campaign').agg({
        'Order_ID': 'count',
        'Total_Amount_INR': 'sum',
        'Customer_ID': 'nunique'
    }).reset_index()
    campaign_performance.columns = ['Campaign', 'Orders', 'Revenue', 'Customers']
    campaign_performance['ROI'] = campaign_performance['Revenue'] / campaign_performance['Orders']
    campaign_performance = campaign_performance.sort_values('Revenue', ascending=False).head(15)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Campaign Revenue', 'Campaign Orders'),
        specs=[[{'type':'bar'}, {'type':'bar'}]]
    )
    
    fig.add_trace(
        go.Bar(x=campaign_performance['Campaign'], y=campaign_performance['Revenue'],
               marker_color='#4CAF50', name='Revenue',
               text=[f"₹{x/1000:.0f}K" for x in campaign_performance['Revenue']],
               textposition='auto'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=campaign_performance['Campaign'], y=campaign_performance['Orders'],
               marker_color='#2196F3', name='Orders',
               text=campaign_performance['Orders'],
               textposition='auto'),
        row=1, col=2
    )
    
    fig.update_layout(
        height=450,
        title_text="Marketing Campaign Performance",
        title_font=dict(size=18, color='#1f2937'),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig.update_xaxes(tickangle=-45, showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Referral source performance
    col1, col2 = st.columns(2)
    
    with col1:
        referral_data = filtered_df.groupby('Referral_Source').agg({
            'Order_ID': 'count',
            'Total_Amount_INR': 'sum'
        }).reset_index()
        referral_data.columns = ['Source', 'Orders', 'Revenue']
        referral_data['AOV'] = referral_data['Revenue'] / referral_data['Orders']
        referral_data = referral_data.sort_values('Revenue', ascending=False)
        
        fig = go.Figure(data=[go.Bar(
            x=referral_data['Source'],
            y=referral_data['Revenue'],
            marker=dict(
                color=referral_data['Revenue'],
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title="Revenue (₹)")
            ),
            text=[f"₹{x/1000:.0f}K" for x in referral_data['Revenue']],
            textposition='auto',
            hovertemplate='%{x}<br>Revenue: ₹%{y:,.0f}<br>AOV: ₹' + 
                         referral_data['AOV'].apply(lambda x: f'{x:,.0f}').astype(str) + '<extra></extra>'
        )])
        
        fig.update_layout(
            title="Revenue by Referral Source",
            title_font=dict(size=18, color='#1f2937'),
            height=400,
            xaxis_title="Referral Source",
            yaxis_title="Revenue (₹)",
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        fig.update_xaxes(tickangle=-45, showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Conversion funnel
        funnel_data = pd.DataFrame({
            'Stage': ['Total Orders', 'Pending Payment', 'Completed Payment', 'In Transit', 'Delivered'],
            'Count': [
                len(filtered_df),
                len(filtered_df[filtered_df['Payment_Status'] == 'Pending']),
                len(filtered_df[filtered_df['Payment_Status'] == 'Completed']),
                len(filtered_df[filtered_df['Order_Status'] == 'In Transit']),
                len(filtered_df[filtered_df['Order_Status'] == 'Delivered'])
            ]
        })
        
        fig = go.Figure(go.Funnel(
            y=funnel_data['Stage'],
            x=funnel_data['Count'],
            textposition="inside",
            textinfo="value+percent initial",
            marker=dict(color=['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#4CAF50'])
        ))
        
        fig.update_layout(
            title="Order Conversion Funnel",
            title_font=dict(size=18, color='#1f2937'),
            height=400,
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Delivery performance
    st.subheader("Delivery Performance Analysis")
    
    if len(delivered_df) > 0:
        delivery_bins = pd.cut(delivered_df['Days_to_Deliver'].astype(float), 
                              bins=[0, 3, 5, 7, 10], 
                              labels=['0-3 days', '3-5 days', '5-7 days', '7+ days'])
        delivery_dist = delivery_bins.value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure(data=[go.Bar(
                x=delivery_dist.index,
                y=delivery_dist.values,
                marker=dict(
                    color=['#4CAF50', '#8BC34A', '#FF9800', '#f44336']
                ),
                text=delivery_dist.values,
                textposition='auto'
            )])
            
            fig.update_layout(
                title="Delivery Time Distribution",
                title_font=dict(size=18, color='#1f2937'),
                height=400,
                xaxis_title="Delivery Time",
                yaxis_title="Number of Orders",
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Delivery performance by shipping method
            delivery_by_method = delivered_df.groupby('Shipping_Method')['Days_to_Deliver'].apply(
                lambda x: pd.Series({
                    'Avg': x.astype(float).mean(),
                    'Min': x.astype(float).min(),
                    'Max': x.astype(float).max()
                })
            ).reset_index()
            
            fig = go.Figure()
            
            for method in delivery_by_method['Shipping_Method'].unique():
                method_data = delivery_by_method[delivery_by_method['Shipping_Method'] == method]
                fig.add_trace(go.Box(
                    y=delivered_df[delivered_df['Shipping_Method'] == method]['Days_to_Deliver'].astype(float),
                    name=method,
                    marker_color='#4CAF50' if method == 'Express' else '#2196F3'
                ))
            
            fig.update_layout(
                title="Delivery Time by Shipping Method",
                title_font=dict(size=18, color='#1f2937'),
                height=400,
                yaxis_title="Delivery Days",
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=True
            )
            fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Performance scorecard
    st.subheader("Overall Performance Scorecard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Sales performance
        target_revenue = 1000000  # Example target
        revenue_achievement = (total_revenue / target_revenue) * 100
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=revenue_achievement,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Revenue Achievement %", 'font': {'size': 16}},
            delta={'reference': 100},
            gauge={
                'axis': {'range': [None, 150]},
                'bar': {'color': "#4CAF50"},
                'steps': [
                    {'range': [0, 50], 'color': "#ffebee"},
                    {'range': [50, 75], 'color': "#fff9c4"},
                    {'range': [75, 100], 'color': "#c8e6c9"},
                    {'range': [100, 150], 'color': "#4CAF50"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 100
                }
            }
        ))
        
        fig.update_layout(height=300, paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Customer satisfaction
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=customer_satisfaction,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Customer Satisfaction %", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#2196F3"},
                'steps': [
                    {'range': [0, 60], 'color': "#ffebee"},
                    {'range': [60, 80], 'color': "#fff9c4"},
                    {'range': [80, 100], 'color': "#c8e6c9"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        
        fig.update_layout(height=300, paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Order fulfillment
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=order_fulfillment,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Order Fulfillment %", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#FF9800"},
                'steps': [
                    {'range': [0, 70], 'color': "#ffebee"},
                    {'range': [70, 90], 'color': "#fff9c4"},
                    {'range': [90, 100], 'color': "#c8e6c9"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 95
                }
            }
        ))
        
        fig.update_layout(height=300, paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Sales Analytics Dashboard</strong> | Last Updated: October 2025</p>
    <p>Built with Streamlit & Plotly | Data-Driven Business Intelligence</p>
</div>
""", unsafe_allow_html=True)
print("✅ Streamlit app fully loaded and ready.", flush=True)

