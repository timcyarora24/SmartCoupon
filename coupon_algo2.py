
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class PersonalizedCouponML:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.accuracy_metrics = {}
        
    def load_and_process_data(self, data_path="./"):
        """Load and process all CSV files"""
        print(" Loading and processing data...")
        
        # Load all datasets
        datasets = {
            'customer_data': pd.read_csv("C:\personalized_coupon_algo\ecommerce_data_customers.csv"), 
            'data_transaction': pd.read_csv("C:\personalized_coupon_algo\ecommerce_data_transactions.csv"),
            'website_usage_behaviour': pd.read_csv("C:\personalized_coupon_algo\ecommerce_data_website_behavior.csv"),
            'coupon_data': pd.read_csv("C:\personalized_coupon_algo\ecommerce_data_coupon_history.csv")
        }
        
        # Process customer data
        customers_df = datasets['customer_data'].copy()
        customers_df['age'] = pd.to_numeric(customers_df['age'], errors='coerce')
        customers_df['family_size'] = pd.to_numeric(customers_df['family_size'], errors='coerce')
        customers_df['registration_date'] = pd.to_datetime(customers_df['registration_date'], errors='coerce')
        customers_df['is_premium_member'] = customers_df['is_premium_member'].astype(bool)
        
        # Process transaction data
        transactions = datasets['data_transaction'].copy()
        transactions['order_date'] = pd.to_datetime(transactions['order_date'], errors='coerce')
        transactions['gross_amount'] = pd.to_numeric(transactions['gross_amount'], errors='coerce')
        transactions['discount_percent'] = pd.to_numeric(transactions['discount_percent'], errors='coerce')
        transactions['final_amount'] = pd.to_numeric(transactions['final_amount'], errors='coerce')
        
        # Calculate customer transaction features
        transaction_features = transactions.groupby('customer_id').agg({
            'order_id': 'count',
            'gross_amount': ['sum', 'mean'],
            'discount_percent': 'mean',
            'final_amount': ['sum', 'mean'],
            'total_items': 'sum',
            'delivery_days': 'mean'
        }).round(2)
        
        transaction_features.columns = [
            'total_orders', 'total_spent', 'avg_order_value', 'avg_discount_used',
            'total_final_amount', 'avg_final_amount', 'total_items_bought', 'avg_delivery_days'
        ]
        
        # Calculate recency, frequency, monetary (RFM) features
        transaction_features['recency'] = (
            pd.Timestamp.now() - transactions.groupby('customer_id')['order_date'].max()
        ).dt.days
        
        transaction_features['frequency'] = transaction_features['total_orders']
        transaction_features['monetary'] = transaction_features['total_spent']
        
        # Process website behavior
        behavior = datasets['website_usage_behaviour'].copy()
        behavior.columns = behavior.columns.str.strip()
        print("📄 Columns:", behavior.columns.tolist())
        behavior['pages_visited'] = pd.to_numeric(behavior['pages_visited'], errors='coerce')
        behavior['session_duration_seconds'] = pd.to_numeric(behavior['session_duration_seconds'], errors='coerce')
        behavior['items_viewed'] = pd.to_numeric(behavior['items_viewed'], errors='coerce')
        behavior['items_added_to_cart'] = pd.to_numeric(behavior['items_added_to_cart'], errors='coerce')
        
        # Calculate behavior features
        behavior_features = behavior.groupby('customer_id').agg({
            'session_id': 'count',
            'pages_visited': 'sum',
            'session_duration_seconds': 'mean',
            'items_viewed': 'sum',
            'items_added_to_cart': 'sum'
        }).round(2)
        
        behavior_features.columns = [
            'total_sessions', 'total_pages_visited', 'avg_session_duration',
            'total_items_viewed', 'total_items_added_to_cart'
        ]
        
        # Calculate engagement metrics
        behavior_features['engagement_score'] = (
            behavior_features['total_items_viewed'] / behavior_features['total_sessions']
        )
        behavior_features['conversion_rate'] = (
            behavior_features['total_items_added_to_cart'] / behavior_features['total_items_viewed']
        )
        
        # Process coupon data to create target variables
        coupons = datasets['coupon_data'].copy()
        coupons['discount_percent'] = pd.to_numeric(coupons['discount_percent'], errors='coerce')
        coupons['redeemed'] = coupons['redeemed'].astype(bool)
        
        # Calculate coupon effectiveness for each customer
        coupon_features = coupons.groupby('customer_id').agg({
            'coupon_id': 'count',
            'discount_percent': 'mean',
            'redeemed': 'mean'
        }).round(3)
        
        coupon_features.columns = ['total_coupons_received', 'avg_coupon_discount', 'redemption_rate']
        
        # Merge all features
        master_df = customers_df.merge(transaction_features, on='customer_id', how='left')
        master_df = master_df.merge(behavior_features, on='customer_id', how='left')
        master_df = master_df.merge(coupon_features, on='customer_id', how='left')
        
        # Fill missing values
        master_df = master_df.fillna(0)
        
        return master_df, coupons
    
    def create_customer_segments(self, df):
        """Create customer segments using K-means clustering"""
        print(" Creating customer segments...")
        
        # Features for clustering
        cluster_features = ['total_spent', 'total_orders', 'avg_order_value', 'recency', 
                          'total_sessions', 'engagement_score', 'redemption_rate']
        
        # Prepare data for clustering
        cluster_data = df[cluster_features].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        cluster_data_scaled = scaler.fit_transform(cluster_data)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        df['customer_segment'] = kmeans.fit_predict(cluster_data_scaled)
        
        # Map segments to meaningful names
        segment_mapping = {
            0: 'Low Value',
            1: 'Medium Value', 
            2: 'High Value',
            3: 'VIP'
        }
        
        df['customer_segment'] = df['customer_segment'].map(segment_mapping)
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for ML models"""
        print("🔧 Preparing features for ML models...")
        
        # Select features for prediction
        feature_columns = [
            'age', 'family_size', 'is_premium_member', 'total_orders', 'total_spent',
            'avg_order_value', 'avg_discount_used', 'recency', 'frequency', 'monetary',
            'total_sessions', 'total_pages_visited', 'avg_session_duration',
            'engagement_score', 'conversion_rate', 'total_coupons_received', 'redemption_rate'
        ]
        
        # Encode categorical variables
        categorical_features = ['gender', 'city', 'income_bracket', 'education', 
                              'marital_status', 'profession', 'customer_segment']
        
        df_encoded = df.copy()
        
        for col in categorical_features:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = le
                feature_columns.append(col + '_encoded')
        
        # Create target variables
        # 1. Optimal discount percentage
        df_encoded['optimal_discount'] = self.calculate_optimal_discount(df_encoded)
        
        # 2. Redemption probability
        df_encoded['redemption_probability'] = df_encoded['redemption_rate'].fillna(0.3)
        
        # 3. Expected revenue
        df_encoded['expected_revenue'] = (
            df_encoded['avg_order_value'] * df_encoded['redemption_probability'] * 
            (1 - df_encoded['optimal_discount'] / 100)
        )
        
        return df_encoded, feature_columns
    
    def calculate_optimal_discount(self, df):
        """Calculate optimal discount based on customer characteristics"""
        discount = np.zeros(len(df))
        
        # Base discount on customer segment
        segment_discount = {
            'Low Value': 15,
            'Medium Value': 10,
            'High Value': 8,
            'VIP': 5
        }
        
        for segment, base_discount in segment_discount.items():
            mask = df['customer_segment'] == segment
            discount[mask] = base_discount
        
        # Adjust based on recency (more recent = lower discount needed)
        recency_adjustment = np.clip(df['recency'] / 30, 0, 10)  # Max 10% increase
        discount += recency_adjustment
        
        # Adjust based on redemption rate (lower rate = higher discount)
        redemption_adjustment = (1 - df['redemption_rate']) * 5
        discount += redemption_adjustment
        
        # Adjust based on engagement (higher engagement = lower discount)
        engagement_adjustment = -df['engagement_score'] * 2
        discount += engagement_adjustment
        
        # Ensure discount is within reasonable bounds
        discount = np.clip(discount, 5, 50)
        
        return discount


    def train_models(self, df, feature_columns):
        """Train machine learning models"""
        print("🤖 Training machine learning models...")

        X = df[feature_columns].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['features'] = scaler

        targets = ['optimal_discount', 'redemption_probability', 'expected_revenue']

        for target in targets:
            print(f"Training models for {target}...")
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'Linear Regression': LinearRegression()
            }

            best_model = None
            best_score = -np.inf

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                cv_score = cross_val_score(model, X_scaled, y, cv=5, scoring='r2').mean()

                if target not in self.accuracy_metrics:
                    self.accuracy_metrics[target] = {}

                self.accuracy_metrics[target][name] = {
                    'MAE': mae,
                    'MSE': mse,
                    'R2': r2,
                    'CV_Score': cv_score
                }

                if r2 > best_score:
                    best_score = r2
                    best_model = model

            self.models[target] = best_model

            # Optional: print trained models
            print(f"Model stored for '{target}'")

            if hasattr(best_model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_columns,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.feature_importance[target] = importance_df
    
    def generate_personalized_coupons(self, df, feature_columns):
        """Generate personalized coupons for customers"""
        print("Generating personalized coupons...")
        
        # Prepare features
        X = df[feature_columns].fillna(0)
        X_scaled = self.scalers['features'].transform(X)
        
        # Predict optimal values
        predictions = {}
        for target in ['optimal_discount', 'redemption_probability', 'expected_revenue']:
            predictions[target] = self.models[target].predict(X_scaled)
        
        # Create recommendations dataframe
        recommendations = pd.DataFrame({
            'customer_id': df['customer_id'],
            'customer_segment': df['customer_segment'],
            'predicted_discount': predictions['optimal_discount'],
            'predicted_redemption_prob': predictions['redemption_probability'],
            'predicted_revenue': predictions['expected_revenue'],
            'current_redemption_rate': df['redemption_rate'],
            'total_spent': df['total_spent'],
            'avg_order_value': df['avg_order_value']
        })
        
        # Generate coupon types based on predictions
        recommendations['coupon_type'] = recommendations.apply(self.assign_coupon_type, axis=1)
        
        # Round predictions
        recommendations['predicted_discount'] = recommendations['predicted_discount'].round(0)
        recommendations['predicted_redemption_prob'] = recommendations['predicted_redemption_prob'].round(3)
        recommendations['predicted_revenue'] = recommendations['predicted_revenue'].round(2)
        
        return recommendations
    
    def assign_coupon_type(self, row):
        """Assign coupon type based on customer characteristics"""
        discount = row['predicted_discount']
        segment = row['customer_segment']
        
        if segment == 'VIP':
            return 'Exclusive VIP Offer'
        elif discount >= 20:
            return 'High Value Discount'
        elif discount >= 15:
            return 'Standard Discount'
        elif discount >= 10:
            return 'Premium Discount'
        else:
            return 'Loyalty Reward'
    
    def visualize_results(self, df, recommendations):
        """Create comprehensive visualizations"""
        print("Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Customer Segment Distribution
        plt.subplot(3, 4, 1)
        segment_counts = df['customer_segment'].value_counts()
        plt.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
        plt.title('Customer Segment Distribution')
        
        # 2. Discount Distribution by Segment
        plt.subplot(3, 4, 2)
        sns.boxplot(data=recommendations, x='customer_segment', y='predicted_discount')
        plt.title('Predicted Discount by Segment')
        plt.xticks(rotation=45)
        
        # 3. Redemption Probability Distribution
        plt.subplot(3, 4, 3)
        plt.hist(recommendations['predicted_redemption_prob'], bins=30, alpha=0.7)
        plt.title('Redemption Probability Distribution')
        plt.xlabel('Predicted Redemption Probability')
        
        # 4. Expected Revenue by Segment
        plt.subplot(3, 4, 4)
        sns.barplot(data=recommendations, x='customer_segment', y='predicted_revenue')
        plt.title('Expected Revenue by Segment')
        plt.xticks(rotation=45)
        
        # 5. Coupon Type Distribution
        plt.subplot(3, 4, 5)
        coupon_counts = recommendations['coupon_type'].value_counts()
        plt.pie(coupon_counts.values, labels=coupon_counts.index, autopct='%1.1f%%')
        plt.title('Coupon Type Distribution')
        
        # 6. Feature Importance (for discount prediction)
        plt.subplot(3, 4, 6)
        if 'optimal_discount' in self.feature_importance:
            importance_df = self.feature_importance['optimal_discount'].head(10)
            plt.barh(importance_df['feature'], importance_df['importance'])
            plt.title('Top 10 Feature Importance (Discount)')
        
        # 7. Correlation Heatmap
        plt.subplot(3, 4, 7)
        corr_features = ['predicted_discount', 'predicted_redemption_prob', 'predicted_revenue', 
                        'total_spent', 'avg_order_value', 'current_redemption_rate']
        corr_data = recommendations[corr_features].corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        
        # 8. Actual vs Predicted Redemption Rate
        plt.subplot(3, 4, 8)
        plt.scatter(recommendations['current_redemption_rate'], recommendations['predicted_redemption_prob'])
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('Actual Redemption Rate')
        plt.ylabel('Predicted Redemption Probability')
        plt.title('Actual vs Predicted Redemption')
        
        # 9. Revenue Distribution
        plt.subplot(3, 4, 9)
        plt.hist(recommendations['predicted_revenue'], bins=30, alpha=0.7)
        plt.title('Expected Revenue Distribution')
        plt.xlabel('Predicted Revenue')
        
        # 10. Segment Performance
        plt.subplot(3, 4, 10)
        segment_perf = recommendations.groupby('customer_segment').agg({
            'predicted_revenue': 'sum',
            'predicted_redemption_prob': 'mean'
        })
        
        ax = plt.gca()
        ax2 = ax.twinx()
        
        bars = ax.bar(segment_perf.index, segment_perf['predicted_revenue'], alpha=0.7)
        line = ax2.plot(segment_perf.index, segment_perf['predicted_redemption_prob'], 'ro-', linewidth=2)
        
        ax.set_ylabel('Total Expected Revenue')
        ax2.set_ylabel('Avg Redemption Probability')
        plt.title('Segment Performance')
        plt.xticks(rotation=45)
        
        # 11. Model Accuracy Comparison
        plt.subplot(3, 4, 11)
        if 'optimal_discount' in self.accuracy_metrics:
            models = list(self.accuracy_metrics['optimal_discount'].keys())
            r2_scores = [self.accuracy_metrics['optimal_discount'][model]['R2'] for model in models]
            
            plt.bar(models, r2_scores)
            plt.title('Model R² Scores (Discount Prediction)')
            plt.ylabel('R² Score')
            plt.xticks(rotation=45)
        
        # 12. Business Impact
        plt.subplot(3, 4, 12)
        total_revenue = recommendations['predicted_revenue'].sum()
        total_customers = len(recommendations)
        avg_discount = recommendations['predicted_discount'].mean()
        
        metrics = ['Total Revenue (K)', 'Total Customers (K)', 'Avg Discount (%)']
        values = [total_revenue/1000, total_customers/1000, avg_discount]
        
        plt.bar(metrics, values)
        plt.title('Business Impact Summary')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def print_accuracy_report(self):
        """Print detailed accuracy report"""
        print("\n" + "="*60)
        print("🎯 MODEL ACCURACY REPORT")
        print("="*60)
        
        for target, models in self.accuracy_metrics.items():
            print(f"\n{target.upper()} PREDICTION:")
            print("-" * 40)
            
            for model_name, metrics in models.items():
                print(f"\n{model_name}:")
                print(f"  • R² Score: {metrics['R2']:.3f}")
                print(f"  • Cross-Val Score: {metrics['CV_Score']:.3f}")
                print(f"  • Mean Absolute Error: {metrics['MAE']:.3f}")
                print(f"  • Mean Squared Error: {metrics['MSE']:.3f}")
                
                # Interpret accuracy
                if metrics['R2'] > 0.8:
                    print(f"  • Accuracy: Excellent ")
                elif metrics['R2'] > 0.6:
                    print(f"  • Accuracy: Good ")
                elif metrics['R2'] > 0.4:
                    print(f"  • Accuracy: Fair ")
                else:
                    print(f"  • Accuracy: Poor ")
    
    def generate_business_report(self, recommendations):
        """Generate business impact report"""
        print("\n" + "="*60)
        print("💼 BUSINESS IMPACT REPORT")
        print("="*60)
        
        # Overall metrics
        total_customers = len(recommendations)
        total_expected_revenue = recommendations['predicted_revenue'].sum()
        avg_discount = recommendations['predicted_discount'].mean()
        avg_redemption_prob = recommendations['predicted_redemption_prob'].mean()
        
        print(f"\n OVERALL METRICS:")
        print(f"  • Total Customers: {total_customers:,}")
        print(f"  • Expected Revenue: ₹{total_expected_revenue:,.2f}")
        print(f"  • Average Discount: {avg_discount:.1f}%")
        print(f"  • Average Redemption Probability: {avg_redemption_prob:.3f}")
        
        # Segment-wise performance
        print(f"\n SEGMENT-WISE PERFORMANCE:")
        segment_summary = recommendations.groupby('customer_segment').agg({
            'customer_id': 'count',
            'predicted_revenue': 'sum',
            'predicted_discount': 'mean',
            'predicted_redemption_prob': 'mean'
        }).round(2)
        
        segment_summary.columns = ['Customer Count', 'Expected Revenue', 'Avg Discount', 'Avg Redemption Prob']
        print(segment_summary)
        
        # Coupon type analysis
        print(f"\nCOUPON TYPE ANALYSIS:")
        coupon_analysis = recommendations.groupby('coupon_type').agg({
            'customer_id': 'count',
            'predicted_revenue': 'sum',
            'predicted_discount': 'mean'
        }).round(2)
        
        coupon_analysis.columns = ['Customer Count', 'Expected Revenue', 'Avg Discount']
        print(coupon_analysis)
        
        return {
            'total_customers': total_customers,
            'total_expected_revenue': total_expected_revenue,
            'avg_discount': avg_discount,
            'avg_redemption_prob': avg_redemption_prob,
            'segment_summary': segment_summary,
            'coupon_analysis': coupon_analysis
        }
    
    def run_complete_pipeline(self, data_path="./"):
        """Run the complete ML pipeline"""
        print("Starting Personalized Coupon Generation ML Pipeline")
        print("="*60)
        
        # Step 1: Load and process data
        df, coupons = self.load_and_process_data(data_path)
        
        # Step 2: Create customer segments
        df = self.create_customer_segments(df)
        
        # Step 3: Prepare features
        df, feature_columns = self.prepare_features(df)
        
        # Step 4: Train models
        self.train_models(df, feature_columns)
        
        # Step 5: Generate recommendations
        recommendations = self.generate_personalized_coupons(df, feature_columns)
        
        # Step 6: Create visualizations
        fig = self.visualize_results(df, recommendations)
        
        # Step 7: Print reports
        self.print_accuracy_report()
        business_report = self.generate_business_report(recommendations)
        
        print("\nPipeline completed successfully!")
        
        return df, recommendations, business_report, fig


if __name__ == "__main__":
    # Initialize the system
    coupon_system = PersonalizedCouponML()
    
    # Run the complete pipeline
    customers_df, recommendations, business_report, visualization = coupon_system.run_complete_pipeline("./")
    
    # Display sample recommendations
    print("\n SAMPLE RECOMMENDATIONS:")
    print(recommendations.head(10))
    
    # Save results
    recommendations.to_csv('personalized_coupon_recommendations.csv', index=False)
    print("\n💾 Recommendations saved to 'personalized_coupon_recommendations.csv'")