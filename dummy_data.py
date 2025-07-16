import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

class RealisticDataGenerator:
    def __init__(self):
        self.cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Pune', 'Kolkata', 'Ahmedabad', 'Jaipur', 'Lucknow']
        self.product_categories = ['Groceries', 'Electronics', 'Fashion', 'Home & Garden', 'Beauty & Personal Care', 
                                 'Sports & Fitness', 'Books & Media', 'Toys & Games', 'Automotive', 'Health & Wellness']
        self.products = {
            'Groceries': ['Rice', 'Wheat', 'Milk', 'Eggs', 'Vegetables', 'Fruits', 'Snacks', 'Beverages', 'Oil', 'Spices'],
            'Electronics': ['Smartphone', 'Laptop', 'Headphones', 'Smartwatch', 'Tablet', 'Camera', 'TV', 'Speaker', 'Gaming Console', 'Accessories'],
            'Fashion': ['Shirts', 'Pants', 'Dresses', 'Shoes', 'Bags', 'Jewelry', 'Watches', 'Sunglasses', 'Ethnic Wear', 'Innerwear'],
            'Home & Garden': ['Furniture', 'Bedding', 'Kitchen Appliances', 'Decoration', 'Cleaning Supplies', 'Tools', 'Plants', 'Lighting', 'Storage', 'Curtains'],
            'Beauty & Personal Care': ['Skincare', 'Makeup', 'Haircare', 'Fragrances', 'Bath Products', 'Oral Care', 'Grooming', 'Nail Care', 'Body Care', 'Men Care'],
            'Sports & Fitness': ['Gym Equipment', 'Sports Shoes', 'Yoga Mat', 'Protein Supplements', 'Sportswear', 'Outdoor Gear', 'Fitness Tracker', 'Bicycle', 'Swimming', 'Team Sports'],
            'Books & Media': ['Fiction Books', 'Non-Fiction', 'Educational', 'Comics', 'Magazines', 'E-books', 'Audiobooks', 'Movies', 'Music', 'Games'],
            'Toys & Games': ['Action Figures', 'Board Games', 'Educational Toys', 'Remote Control', 'Puzzles', 'Dolls', 'Building Blocks', 'Art Supplies', 'Outdoor Toys', 'Video Games'],
            'Automotive': ['Car Accessories', 'Motorcycle Parts', 'Tires', 'Oil & Lubricants', 'Electronics', 'Cleaning', 'Tools', 'Safety', 'Interior', 'Exterior'],
            'Health & Wellness': ['Vitamins', 'Supplements', 'Medical Devices', 'First Aid', 'Fitness Equipment', 'Massage', 'Ayurvedic', 'Homeopathy', 'Organic Products', 'Health Monitors']
        }
        self.coupon_types = ['Percentage', 'Fixed Amount', 'Buy One Get One', 'Free Shipping', 'Category Specific', 'Minimum Purchase', 'Cashback', 'Loyalty Points']
        
    def generate_customer_demographics(self, n_customers=5000):
        customers = []
        
        for i in range(n_customers):
            customer_id = f"CUST_{i+1:06d}"
            
            age = np.random.choice([
                np.random.randint(18, 25),
                np.random.randint(25, 35),
                np.random.randint(35, 45),
                np.random.randint(45, 60),
                np.random.randint(60, 80)
            ], p=[0.15, 0.35, 0.25, 0.20, 0.05])
            
            gender = np.random.choice(['Male', 'Female', 'Other'], p=[0.52, 0.47, 0.01])
            
            city = np.random.choice(self.cities, p=[0.20, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.13])
            
            income_bracket = np.random.choice(['Low', 'Medium', 'High', 'Premium'], p=[0.30, 0.45, 0.20, 0.05])
            
            education = np.random.choice(['High School', 'Graduate', 'Post Graduate', 'Professional'], p=[0.25, 0.40, 0.25, 0.10])
            
            marital_status = np.random.choice(['Single', 'Married', 'Divorced'], p=[0.35, 0.60, 0.05])
            family_size = np.random.randint(1, 6) if marital_status == 'Married' else np.random.randint(1, 3)
            
            profession = np.random.choice([
                'Student', 'IT Professional', 'Business', 'Government', 'Healthcare', 
                'Education', 'Retired', 'Homemaker', 'Manufacturing', 'Service', 'Other'
            ], p=[0.08, 0.22, 0.15, 0.12, 0.08, 0.07, 0.05, 0.10, 0.08, 0.03, 0.02])
            
            registration_date = datetime.now() - timedelta(days=np.random.randint(1, 1460))
            
            mobile_number = f"+91{np.random.randint(7000000000, 9999999999)}"
            email_domain = np.random.choice(['gmail.com', 'yahoo.com', 'outlook.com', 'company.com'], p=[0.60, 0.20, 0.15, 0.05])
            email = f"{customer_id.lower()}@{email_domain}"
            
            customers.append({
                'customer_id': customer_id,
                'age': age,
                'gender': gender,
                'city': city,
                'income_bracket': income_bracket,
                'education': education,
                'marital_status': marital_status,
                'family_size': family_size,
                'profession': profession,
                'registration_date': registration_date,
                'mobile_number': mobile_number,
                'email': email,
                'is_premium_member': np.random.choice([True, False], p=[0.15, 0.85])
            })
        
        return pd.DataFrame(customers)
    
    def generate_transaction_history(self, customers_df, months_history=24):
        transactions = []
        
        for _, customer in customers_df.iterrows():
            customer_behavior = self._determine_customer_behavior(customer)
            
            total_months = min(months_history, (datetime.now() - customer['registration_date']).days // 30)
            
            for month in range(total_months):
                month_orders = np.random.poisson(customer_behavior['orders_per_month'])
                
                for order_num in range(month_orders):
                    order_date = customer['registration_date'] + timedelta(
                        days=month * 30 + np.random.randint(0, 30)
                    )
                    
                    if order_date > datetime.now():
                        continue
                    
                    n_items = max(1, np.random.poisson(customer_behavior['items_per_order']))
                    total_amount = 0
                    order_items = []
                    
                    for item_num in range(n_items):
                        category = np.random.choice(
                            self.product_categories, 
                            p=customer_behavior['category_preferences']
                        )
                        product = np.random.choice(self.products[category])
                        
                        base_price = self._get_product_price(category, product)
                        seasonal_factor = self._get_seasonal_factor(order_date, category)
                        final_price = base_price * seasonal_factor
                        
                        quantity = np.random.choice([1, 2, 3, 4, 5], p=[0.60, 0.25, 0.10, 0.03, 0.02])
                        item_total = final_price * quantity
                        total_amount += item_total
                        
                        order_items.append({
                            'product_name': product,
                            'category': category,
                            'quantity': int(quantity),
                            'unit_price': float(final_price),
                            'total_price': float(item_total)
                        })
                    
                    discount_applied = self._calculate_discount(customer, total_amount)
                    final_amount = total_amount * (1 - discount_applied/100)
                    
                    payment_method = np.random.choice([
                        'Credit Card', 'Debit Card', 'UPI', 'Net Banking', 'Cash on Delivery', 'Digital Wallet'
                    ], p=[0.25, 0.20, 0.30, 0.10, 0.10, 0.05])
                    
                    delivery_days = np.random.choice([1, 2, 3, 4, 5, 7, 10], p=[0.15, 0.30, 0.25, 0.15, 0.10, 0.04, 0.01])
                    
                    order_status = np.random.choice([
                        'Delivered', 'Cancelled', 'Returned', 'Pending'
                    ], p=[0.85, 0.10, 0.03, 0.02])
                    
                    transactions.append({
                        'customer_id': customer['customer_id'],
                        'order_id': f"ORD_{len(transactions)+1:08d}",
                        'order_date': order_date,
                        'total_items': n_items,
                        'gross_amount': total_amount,
                        'discount_percent': discount_applied,
                        'final_amount': final_amount,
                        'payment_method': payment_method,
                        'delivery_days': delivery_days,
                        'order_status': order_status,
                        'primary_category': max(set([item['category'] for item in order_items]), 
                                              key=[item['category'] for item in order_items].count),
                        'order_items_json': json.dumps(order_items)
                    })
        
        return pd.DataFrame(transactions)
    
    def generate_coupon_history(self, customers_df, transactions_df):
        coupon_history = []
        
        for _, customer in customers_df.iterrows():
            customer_transactions = transactions_df[transactions_df['customer_id'] == customer['customer_id']]
            
            if len(customer_transactions) == 0:
                continue
            
            transaction_count = len(customer_transactions)
            base_coupons = max(1, transaction_count // 3)
            n_coupons = np.random.poisson(base_coupons)
            
            for coupon_num in range(n_coupons):
                coupon_date = self._generate_coupon_date(customer, customer_transactions)
                
                coupon_type = np.random.choice(self.coupon_types, p=[0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03])
                
                coupon_details = self._generate_coupon_details(coupon_type, customer, customer_transactions)
                
                redemption_prob = self._calculate_redemption_probability(
                    customer, customer_transactions, coupon_details
                )
                
                redeemed = np.random.random() < redemption_prob
                redemption_date = None
                order_used = None
                
                if redeemed:
                    future_orders = customer_transactions[customer_transactions['order_date'] > coupon_date]
                    if len(future_orders) > 0:
                        suitable_orders = future_orders[future_orders['final_amount'] >= coupon_details['min_purchase']]
                        if len(suitable_orders) > 0:
                            order_used = suitable_orders.iloc[0]['order_id']
                            redemption_date = suitable_orders.iloc[0]['order_date']
                        else:
                            redeemed = False
                    else:
                        redeemed = False
                
                coupon_history.append({
                    'customer_id': customer['customer_id'],
                    'coupon_id': f"CPN_{len(coupon_history)+1:08d}",
                    'coupon_type': coupon_type,
                    'issue_date': coupon_date,
                    'expiry_date': coupon_date + timedelta(days=int(coupon_details['validity_days'])),
                    'discount_percent': coupon_details.get('discount_percent', 0),
                    'discount_amount': coupon_details.get('discount_amount', 0),
                    'min_purchase': coupon_details['min_purchase'],
                    'max_discount': coupon_details.get('max_discount', 0),
                    'applicable_category': coupon_details.get('category', 'All'),
                    'validity_days': coupon_details['validity_days'],
                    'redeemed': redeemed,
                    'redemption_date': redemption_date,
                    'order_used': order_used,
                    'coupon_source': np.random.choice(['Email', 'SMS', 'App Push', 'Website Banner'], p=[0.40, 0.30, 0.20, 0.10])
                })
        
        return pd.DataFrame(coupon_history)
    
    def generate_website_behavior(self, customers_df, sessions_per_customer=50):
        website_sessions = []
        
        for _, customer in customers_df.iterrows():
            n_sessions = max(1, np.random.poisson(sessions_per_customer))
            
            for session_num in range(n_sessions):
                session_date = self._generate_session_date(customer)
                
                device_type = np.random.choice(['Mobile', 'Desktop', 'Tablet'], p=[0.70, 0.25, 0.05])
                platform = np.random.choice(['Android', 'iOS', 'Web'], p=[0.45, 0.25, 0.30])
                
                pages_visited = max(1, np.random.poisson(6))
                session_duration = max(30, np.random.exponential(420))
                
                categories_browsed = np.random.choice(
                    self.product_categories, 
                    size=np.random.randint(1, 4), 
                    replace=False
                ).tolist()
                
                items_viewed = np.random.poisson(pages_visited * 0.7)
                items_added_to_cart = np.random.poisson(items_viewed * 0.1)
                items_added_to_wishlist = np.random.poisson(items_viewed * 0.05)
                
                session_outcome = np.random.choice([
                    'Purchase', 'Cart Abandonment', 'Browsing', 'Bounce'
                ], p=[0.05, 0.15, 0.60, 0.20])
                
                website_sessions.append({
                    'customer_id': customer['customer_id'],
                    'session_id': f"SESS_{len(website_sessions)+1:08d}",
                    'session_date': session_date,
                    'device_type': device_type,
                    'platform': platform,
                    'pages_visited': pages_visited,
                    'session_duration_seconds': int(session_duration),
                    'categories_browsed': ','.join(categories_browsed),
                    'items_viewed': items_viewed,
                    'items_added_to_cart': items_added_to_cart,
                    'items_added_to_wishlist': items_added_to_wishlist,
                    'session_outcome': session_outcome,
                    'referrer_source': np.random.choice(['Direct', 'Google', 'Facebook', 'Email', 'Other'], p=[0.35, 0.30, 0.15, 0.10, 0.10])
                })
        
        return pd.DataFrame(website_sessions)
    
    def _determine_customer_behavior(self, customer):
        base_orders = 2
        
        if customer['age'] < 25:
            age_factor = 0.7
        elif customer['age'] > 55:
            age_factor = 0.8
        else:
            age_factor = 1.0
        
        income_factors = {'Low': 0.5, 'Medium': 1.0, 'High': 1.5, 'Premium': 2.0}
        income_factor = income_factors[customer['income_bracket']]
        
        family_factor = min(customer['family_size'] * 0.4, 1.8)
        
        premium_factor = 1.3 if customer['is_premium_member'] else 1.0
        
        orders_per_month = max(0.5, base_orders * age_factor * income_factor * family_factor * premium_factor)
        items_per_order = max(1, int(2 * family_factor))
        
        category_prefs = [0.1] * len(self.product_categories)
        
        if customer['age'] < 30:
            category_prefs[self.product_categories.index('Electronics')] = 0.20
            category_prefs[self.product_categories.index('Fashion')] = 0.18
        elif customer['age'] > 45:
            category_prefs[self.product_categories.index('Health & Wellness')] = 0.15
            category_prefs[self.product_categories.index('Home & Garden')] = 0.13
        
        if customer['income_bracket'] in ['High', 'Premium']:
            category_prefs[self.product_categories.index('Electronics')] *= 1.5
            category_prefs[self.product_categories.index('Fashion')] *= 1.3
        
        if customer['family_size'] > 3:
            category_prefs[self.product_categories.index('Groceries')] = 0.25
            category_prefs[self.product_categories.index('Home & Garden')] *= 1.2
        
        category_prefs = np.array(category_prefs)
        category_prefs = category_prefs / category_prefs.sum()
        
        return {
            'orders_per_month': orders_per_month,
            'items_per_order': items_per_order,
            'category_preferences': category_prefs
        }
    
    def _get_product_price(self, category, product):
        price_ranges = {
            'Groceries': (20, 800),
            'Electronics': (500, 80000),
            'Fashion': (200, 8000),
            'Home & Garden': (150, 15000),
            'Beauty & Personal Care': (100, 3000),
            'Sports & Fitness': (300, 12000),
            'Books & Media': (50, 1500),
            'Toys & Games': (100, 5000),
            'Automotive': (200, 25000),
            'Health & Wellness': (100, 4000)
        }
        
        min_price, max_price = price_ranges[category]
        
        mean_log = np.log(min_price + (max_price - min_price) * 0.3)
        std_log = 0.8
        price = np.random.lognormal(mean_log, std_log)
        
        return max(min_price, min(price, max_price))
    
    def _get_seasonal_factor(self, order_date, category):
        month = order_date.month
        
        seasonal_factors = {
            'Groceries': 1.0,
            'Electronics': 1.3 if month in [10, 11, 12] else 1.0,
            'Fashion': 1.2 if month in [10, 11, 3, 4] else 1.0,
            'Home & Garden': 1.15 if month in [3, 4, 5, 10, 11] else 1.0,
            'Beauty & Personal Care': 1.1 if month in [10, 11, 12, 2] else 1.0,
            'Sports & Fitness': 1.15 if month in [1, 2, 6, 7] else 1.0,
            'Books & Media': 1.05 if month in [6, 7, 8, 12] else 1.0,
            'Toys & Games': 1.4 if month in [10, 11, 12] else 1.0,
            'Automotive': 1.05 if month in [10, 11, 3, 4] else 1.0,
            'Health & Wellness': 1.1 if month in [1, 2, 6, 7] else 1.0
        }
        
        return seasonal_factors.get(category, 1.0)
    
    def _calculate_discount(self, customer, order_amount):
        base_discount = 0
        
        if customer['is_premium_member']:
            base_discount = 5
        
        if order_amount > 5000:
            base_discount += 10
        elif order_amount > 2000:
            base_discount += 5
        elif order_amount > 1000:
            base_discount += 2
        
        promo_discount = np.random.choice([0, 5, 10, 15, 20], p=[0.50, 0.25, 0.15, 0.07, 0.03])
        
        return min(base_discount + promo_discount, 30)
    
    def _generate_coupon_date(self, customer, transactions):
        if len(transactions) == 0:
            return customer['registration_date'] + timedelta(days=np.random.randint(1, 30))
        
        random_transaction = transactions.sample(1).iloc[0]
        
        offset_days = np.random.randint(-30, 30)
        coupon_date = random_transaction['order_date'] + timedelta(days=offset_days)
        
        coupon_date = max(coupon_date, customer['registration_date'])
        coupon_date = min(coupon_date, datetime.now())
        
        return coupon_date
    
    def _generate_coupon_details(self, coupon_type, customer, transactions):
        avg_order_value = transactions['final_amount'].mean() if len(transactions) > 0 else 1000
        
        if coupon_type == 'Percentage':
            return {
                'discount_percent': np.random.choice([5, 10, 15, 20, 25], p=[0.30, 0.30, 0.20, 0.15, 0.05]),
                'min_purchase': int(avg_order_value * np.random.choice([0.5, 0.8, 1.0, 1.2], p=[0.30, 0.30, 0.25, 0.15])),
                'max_discount': np.random.choice([500, 1000, 2000, 5000], p=[0.40, 0.35, 0.20, 0.05]),
                'validity_days': np.random.choice([7, 15, 30, 60], p=[0.20, 0.40, 0.30, 0.10])
            }
        
        elif coupon_type == 'Fixed Amount':
            return {
                'discount_amount': np.random.choice([50, 100, 200, 500, 1000], p=[0.30, 0.25, 0.25, 0.15, 0.05]),
                'min_purchase': int(avg_order_value * np.random.choice([0.8, 1.0, 1.5, 2.0], p=[0.25, 0.35, 0.25, 0.15])),
                'validity_days': np.random.choice([7, 15, 30, 60], p=[0.25, 0.35, 0.30, 0.10])
            }
        
        elif coupon_type == 'Cashback':
            return {
                'discount_percent': np.random.choice([5, 10, 15], p=[0.50, 0.35, 0.15]),
                'min_purchase': int(avg_order_value * 0.8),
                'max_discount': np.random.choice([200, 500, 1000], p=[0.50, 0.35, 0.15]),
                'validity_days': np.random.choice([15, 30, 60], p=[0.40, 0.40, 0.20])
            }
        
        else:
            return {
                'discount_percent': np.random.choice([10, 15, 20], p=[0.50, 0.30, 0.20]),
                'min_purchase': int(avg_order_value * 0.7),
                'validity_days': np.random.choice([15, 30], p=[0.60, 0.40])
            }
    
    def _calculate_redemption_probability(self, customer, transactions, coupon_details):
        base_prob = 0.25
        
        if len(transactions) > 0:
            avg_order_value = transactions['final_amount'].mean()
            recent_orders = transactions[transactions['order_date'] > datetime.now() - timedelta(days=30)]
            
            if len(recent_orders) > 0:
                base_prob += 0.15
            elif len(transactions[transactions['order_date'] > datetime.now() - timedelta(days=90)]) > 0:
                base_prob += 0.05
            
            if len(transactions) > 10:
                base_prob += 0.10
            elif len(transactions) > 5:
                base_prob += 0.05
            
            if avg_order_value >= coupon_details['min_purchase']:
                base_prob += 0.20
            elif avg_order_value >= coupon_details['min_purchase'] * 0.8:
                base_prob += 0.10
            else:
                base_prob -= 0.15
        
        if customer['age'] < 35:
            base_prob += 0.05
        
        if customer['income_bracket'] in ['High', 'Premium']:
            base_prob += 0.05
        
        if customer['is_premium_member']:
            base_prob += 0.10
        
        if coupon_details.get('discount_percent', 0) > 15:
            base_prob += 0.05
        
        if coupon_details.get('discount_amount', 0) > 200:
            base_prob += 0.05
        
        return max(0.05, min(base_prob, 0.80))
    
    def _generate_session_date(self, customer):
        days_since_registration = (datetime.now() - customer['registration_date']).days
        session_days_ago = np.random.randint(1, min(days_since_registration + 1, 180))
        return datetime.now() - timedelta(days=session_days_ago)
    
    def generate_complete_dataset(self, n_customers=5000):
        print("🔄 Generating realistic customer demographics...")
        customers_df = self.generate_customer_demographics(n_customers)
        
        print("🔄 Generating transaction history...")
        transactions_df = self.generate_transaction_history(customers_df)
        
        print("🔄 Generating coupon history...")
        coupon_history_df = self.generate_coupon_history(customers_df, transactions_df)
        
        print("🔄 Generating website behavior data...")
        website_behavior_df = self.generate_website_behavior(customers_df)
        
        print("✅ Dataset generation complete!")
        print(f"   - Customers: {len(customers_df):,}")
        print(f"   - Transactions: {len(transactions_df):,}")
        print(f"   - Coupon History: {len(coupon_history_df):,}")
        print(f"   - Website Sessions: {len(website_behavior_df):,}")
        
        return customers_df, transactions_df, coupon_history_df, website_behavior_df
    
    def save_datasets(self, customers_df, transactions_df, coupon_history_df, website_behavior_df, prefix="ecommerce_data"):
        customers_df.to_csv(f"{prefix}_customers.csv", index=False)
        transactions_df.to_csv(f"{prefix}_transactions.csv", index=False)
        coupon_history_df.to_csv(f"{prefix}_coupon_history.csv", index=False)
        website_behavior_df.to_csv(f"{prefix}_website_behavior.csv", index=False)
        
        print(f"✅ All datasets saved with prefix: {prefix}")
    
    def generate_summary_report(self, customers_df, transactions_df, coupon_history_df, website_behavior_df):
        print("📊 DATASET SUMMARY REPORT")
        print("=" * 50)
        
        print(f"📈 Customer Demographics:")
        print(f"   Total Customers: {len(customers_df):,}")
        print(f"   Age Distribution: {customers_df['age'].describe()}")
        print(f"   Gender Distribution: {customers_df['gender'].value_counts().to_dict()}")
        print(f"   City Distribution: {customers_df['city'].value_counts().head()}")
        print(f"   Income Brackets: {customers_df['income_bracket'].value_counts().to_dict()}")
        print(f"   Premium Members: {customers_df['is_premium_member'].sum():,} ({customers_df['is_premium_member'].mean():.1%})")
        
        print(f"\n💳 Transaction Analysis:")
        print(f"   Total Transactions: {len(transactions_df):,}")
        print(f"   Total Revenue: ₹{transactions_df['final_amount'].sum():,.2f}")
        print(f"   Average Order Value: ₹{transactions_df['final_amount'].mean():.2f}")
        print(f"   Order Status Distribution: {transactions_df['order_status'].value_counts().to_dict()}")
        print(f"   Payment Method Distribution: {transactions_df['payment_method'].value_counts().to_dict()}")
        
        print(f"\n🎫 Coupon Analysis:")
        print(f"   Total Coupons Issued: {len(coupon_history_df):,}")
        print(f"   Redeemed Coupons: {coupon_history_df['redeemed'].sum():,} ({coupon_history_df['redeemed'].mean():.1%})")
        print(f"   Coupon Types: {coupon_history_df['coupon_type'].value_counts().to_dict()}")
        
        print(f"\n🌐 Website Behavior:")
        print(f"   Total Sessions: {len(website_behavior_df):,}")
        print(f"   Average Session Duration: {website_behavior_df['session_duration_seconds'].mean():.1f} seconds")
        print(f"   Device Distribution: {website_behavior_df['device_type'].value_counts().to_dict()}")
        print(f"   Platform Distribution: {website_behavior_df['platform'].value_counts().to_dict()}")
        print(f"   Session Outcomes: {website_behavior_df['session_outcome'].value_counts().to_dict()}")
        
        print(f"\n📊 Business Insights:")
        avg_customer_ltv = transactions_df.groupby('customer_id')['final_amount'].sum().mean()
        print(f"   Average Customer LTV: ₹{avg_customer_ltv:.2f}")
        
        repeat_customers = transactions_df.groupby('customer_id').size()
        repeat_rate = (repeat_customers > 1).mean()
        print(f"   Repeat Customer Rate: {repeat_rate:.1%}")
        
        monthly_revenue = transactions_df.groupby(transactions_df['order_date'].dt.to_period('M'))['final_amount'].sum()
        print(f"   Monthly Revenue Trend: ₹{monthly_revenue.mean():.2f} average")
        
        print("\n" + "=" * 50)


if __name__ == "__main__":
    generator = RealisticDataGenerator()
    
    customers_df, transactions_df, coupon_history_df, website_behavior_df = generator.generate_complete_dataset(n_customers=5000)
    
    generator.save_datasets(customers_df, transactions_df, coupon_history_df, website_behavior_df)
    
    generator.generate_summary_report(customers_df, transactions_df, coupon_history_df, website_behavior_df)