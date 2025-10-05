# 📊 Sales Analytics Dashboard

A comprehensive, interactive sales analytics dashboard built with Streamlit and Plotly for data-driven business intelligence.

![Dashboard Preview](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red)
![Plotly](https://img.shields.io/badge/Plotly-5.18.0-purple)

## 🌟 Features

### 📈 7 Comprehensive Analysis Tabs

1. **Overview** - Key business metrics, daily trends, and top performers
2. **Sales Analysis** - Channel performance, payment methods, hourly patterns
3. **Product Insights** - Product performance, category analysis, price points
4. **Customer Analytics** - Demographics, behavior, satisfaction metrics
5. **Geographic Analysis** - Location-based insights, regional performance
6. **Revenue Deep Dive** - Financial breakdown, forecasting, revenue trends
7. **Performance Metrics** - KPIs, marketing campaigns, delivery performance

### ✨ Key Capabilities

- 📤 **File Upload** - Easy CSV upload directly in the app
- 🔍 **Interactive Filters** - Date range, category, city, segment, payment method
- 📊 **30+ Visualizations** - Charts, graphs, heatmaps, treemaps, and more
- 🎨 **Professional Design** - Clean UI with custom styling
- ⚡ **Fast Performance** - Optimized with data caching
- 📱 **Responsive Layout** - Works on desktop and mobile

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sales-analytics-dashboard.git
   cd sales-analytics-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard**
   ```bash
   streamlit run sales_dashboard.py
   ```

4. **Upload your data**
   - Open the dashboard in your browser (usually `http://localhost:8501`)
   - Click "Browse files" in the sidebar
   - Upload your CSV file
   - Start analyzing!

## 📁 Project Structure

```
sales-analytics-dashboard/
├── sales_dashboard.py          # Main dashboard application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── sample_data/               # (Optional) Sample dataset
    └── comprehensive_sales_data_sept_oct_2025.csv
```

## 📊 Data Format

Your CSV file should contain the following columns:

### Required Columns:
- **Order Information**: Order_ID, Order_Date, Order_Time, Order_Status
- **Customer Details**: Customer_ID, Customer_Name, Customer_Email, Customer_Phone, Customer_Age, Customer_Gender, Customer_City, Customer_State, Customer_Country, Customer_Zip, Customer_Segment
- **Product Details**: Product_Name, Product_Category, SKU, Quantity, Unit_Price_INR
- **Financial Data**: Discount_Percent, Discount_Amount_INR, Tax_Amount_INR, Total_Amount_INR
- **Payment & Shipping**: Payment_Method, Payment_Status, Shipping_Method, Shipping_Cost_INR
- **Delivery**: Delivery_Date, Days_to_Deliver
- **Customer Feedback**: Customer_Rating, Customer_Review
- **Marketing**: Sales_Channel, Referral_Source, Marketing_Campaign, First_Time_Customer, Customer_Lifetime_Orders

### Sample Data Format:

```csv
Order_ID,Order_Date,Customer_Name,Product_Name,Product_Category,Total_Amount_INR,...
ORD-10001,2025-09-03,John Doe,Whey Protein 1 Kg,Protein Supplements,4858.02,...
```

## 🎨 Dashboard Sections

### 1. Overview Tab
- Total revenue, orders, and customer metrics
- Daily sales trends
- Order status distribution
- Top products and categories

### 2. Sales Analysis Tab
- Discount and tax analysis
- Sales channel comparison (Website vs Mobile App)
- Payment method breakdown
- Hourly sales patterns
- Shipping preferences

### 3. Product Insights Tab
- Product performance tables
- Category revenue treemaps
- Price point analysis
- Units sold rankings
- Multi-metric comparisons

### 4. Customer Analytics Tab
- Age and gender distribution
- Customer segmentation
- New vs repeat customers
- Lifetime value analysis
- Satisfaction ratings

### 5. Geographic Analysis Tab
- City and state performance
- Revenue heatmaps
- Regional comparisons
- Shipping preferences by location

### 6. Revenue Deep Dive Tab
- Revenue components breakdown
- Weekly trends
- Payment status analysis
- Revenue forecasting
- Moving averages

### 7. Performance Metrics Tab
- KPI gauges and scorecards
- Marketing campaign ROI
- Referral source effectiveness
- Conversion funnels
- Delivery performance

## 🛠️ Technologies Used

- **[Streamlit](https://streamlit.io/)** - Web framework for data apps
- **[Plotly](https://plotly.com/)** - Interactive visualizations
- **[Pandas](https://pandas.pydata.org/)** - Data manipulation
- **[NumPy](https://numpy.org/)** - Numerical computing

## 🌐 Deployment

### Deploy to Streamlit Cloud (Free)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy!

### Deploy to Heroku

```bash
# Install Heroku CLI and login
heroku login

# Create Heroku app
heroku create your-app-name

# Deploy
git push heroku main
```

### Deploy to AWS/GCP/Azure

Follow the respective cloud platform's documentation for deploying Streamlit apps.

## 📸 Screenshots

*Add screenshots of your dashboard here*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Visualizations powered by [Plotly](https://plotly.com/)
- Inspired by modern data analytics best practices

## 📧 Contact

For questions or feedback, please open an issue or contact [your.email@example.com](mailto:your.email@example.com)

---

**⭐ If you find this project helpful, please give it a star!**
