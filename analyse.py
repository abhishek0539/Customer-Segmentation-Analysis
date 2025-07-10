import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog
import warnings
from datetime import datetime
import os
from io import BytesIO
import base64
from fpdf import FPDF
from logger import log, get_log_buffer

warnings.filterwarnings("ignore")

class AnalysisPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Customer Segmentation Analysis Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def add_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def add_text(self, text):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, text)
        self.ln(5)

    def add_image(self, image_path, caption):
        self.add_title(caption)
        self.image(image_path, x=10, w=190)
        self.ln(5)

def load_dataset():
    try:
        print("Please select your dataset (CSV file)...")
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            print("No file selected.")
            return None, None
        data = pd.read_csv(file_path)
        print(f"\nLoaded dataset: {os.path.basename(file_path)}")
        return data, file_path
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None


def check_gender_age_columns(data):
    gender_col = next((col for col in data.columns if col.lower() in ['gender', 'sex', 'genre']), None)
    age_col = next((col for col in data.columns if col.lower() in ['age', 'agegroup', 'age_group']), None)
    return gender_col, age_col

def check_product_column(data):
    product_col = next((col for col in data.columns if col.lower() in ['product category', 'stockcode', 'itemcode']), None)
    return product_col

def save_plot(filename):
    plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)
    plt.close()

def compute_rfm(data, pdf):
    try:
        log("Computing RFM analysis...")
        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], errors='coerce')
        if data['InvoiceDate'].isna().all():
            raise ValueError("Invalid or missing InvoiceDate values.")
        if 'Net Amount' not in data.columns:
            data['Net Amount'] = data['Quantity'] * data['UnitPrice']
        current_date = data['InvoiceDate'].max() + pd.Timedelta(days=1)
        rfm = data.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (current_date - x.max()).days,
            'InvoiceNo': 'nunique',
            'Net Amount': 'sum'
        }).reset_index()
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        rfm = rfm[rfm['Monetary'] >= 0]
        log(f"RFM DataFrame:\n{rfm.head()}")
        pdf.add_title("RFM Analysis")
        table_content = "CustomerID\tRecency\tFrequency\tMonetary\n"
        for _, row in rfm.head(5).iterrows():
            table_content += f"{row['CustomerID']}\t{row['Recency']}\t{row['Frequency']}\t{row['Monetary']:.2f}\n"
        pdf.add_text(table_content)
        return rfm.dropna()
    except Exception as e:
        log(f"Error computing RFM: {e}")
        return None

def assign_rfm_tiles(rfm):
    try:
        log("Assigning RFM tiles...")
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1], duplicates='drop')
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4], duplicates='drop')
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4], duplicates='drop')
        rfm['RFMScore'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

        def rfm_segment(row):
            if row['RFMScore'] in ['444', '434', '443', '344']: return '⭐ Best Customers'
            elif row['R_Score'] >= 3: return '✅ Loyal Customers'
            elif row['F_Score'] <= 2 and row['R_Score'] <= 2: return '😟 At Risk'
            elif row['M_Score'] <= 2: return '📉 Low Spend'
            else: return '🙂 Regulars'

        rfm['RFM_Tile'] = rfm.apply(rfm_segment, axis=1)
        return rfm
    except Exception as e:
        log(f"Error in assigning RFM tiles: {e}")
        return rfm

def churn_prediction(rfm):
    try:
        log("Predicting churn risk (RFM)...")
        rfm['Churn_Risk'] = rfm.apply(
            lambda row: '⚠️ High' if row['Recency'] > rfm['Recency'].quantile(0.75) and
            row['Frequency'] < rfm['Frequency'].quantile(0.25) and
            row['Monetary'] < rfm['Monetary'].median() else 'Low', axis=1)
        return rfm
    except Exception as e:
        log(f"Error in churn prediction (RFM): {e}")
        return rfm

def churn_prediction_non_rfm(data, numeric_cols):
    try:
        log("Predicting churn risk (Non-RFM)...")
        # Identify spending-related column (e.g., Spending Score)
        spending_col = next((col for col in numeric_cols if 'spending' in col.lower() or 'score' in col.lower()), None)
        if not spending_col:
            log("No spending-related column found for churn prediction. Using first numeric column.")
            spending_col = numeric_cols[0] if numeric_cols else None

        if spending_col:
            # High churn risk: low spending score (below 25th percentile)
            data['Churn_Risk'] = data[spending_col].apply(
                lambda x: 'High' if x < data[spending_col].quantile(0.25) else 'Low')
        else:
            # Fallback: use median of first numeric column if no spending column
            data['Churn_Risk'] = 'Low'  # Default if no suitable column

        # Visualize churn risk distribution
        plt.figure(figsize=(8, 5))
        sns.countplot(x='Churn_Risk', data=data, palette='Set2')
        plt.title('Churn Risk Distribution')
        save_plot("churn_risk_distribution.png")
        return data
    except Exception as e:
        log(f"Error in churn prediction (Non-RFM): {e}")
        data['Churn_Risk'] = 'Low'  # Fallback
        return data

def gender_age_analysis(data, gender_col, age_col, pdf):
    try:
        log("Performing gender/age analysis...")
        # Non-date-dependent plots
        if gender_col:
            plt.figure(figsize=(10, 5))
            sns.countplot(x=gender_col, data=data, palette='Set2')
            plt.title('Gender Distribution')
            plt.xticks(rotation=45)
            save_plot("gender_distribution.png")
            pdf.add_image("gender_distribution.png", "Gender Distribution")
            #plt.show()

        if age_col:
            plt.figure(figsize=(10, 5))
            sns.histplot(data[age_col], kde=True, color='#66b3ff')
            plt.title('Age Distribution')
            plt.xlabel(age_col)
            plt.xticks(rotation=45)
            save_plot("age_distribution.png")
            pdf.add_image("age_distribution.png", "Age Distribution")
            #plt.show()

        # Date-dependent plots
        if 'InvoiceDate' in data.columns and 'Net Amount' in data.columns:
            data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], errors='coerce')
            if data['InvoiceDate'].isna().all():
                raise ValueError("Invalid or missing InvoiceDate values.")
            data['DayType'] = data['InvoiceDate'].dt.weekday.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

            if gender_col:
                plt.figure(figsize=(8, 8))
                gender_spend = data.groupby(gender_col)['Net Amount'].sum()
                plt.pie(gender_spend, labels=gender_spend.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999', '#99ff99'])
                plt.title('Spending Distribution by Gender')
                save_plot("gender_spend.png")
                pdf.add_image("gender_spend.png", "Spending Distribution by Gender")
                #plt.show()

                plt.figure(figsize=(10, 5))
                sns.countplot(x=gender_col, hue='DayType', data=data, palette='Set2')
                plt.title('Transaction Frequency by Gender and Day Type')
                save_plot("gender_transaction_freq.png")
                pdf.add_image("gender_transaction_freq.png", "Transaction Frequency by Gender and Day Type")
                #plt.show()

            if age_col:
                plt.figure(figsize=(10, 5))
                age_spend = data.groupby(age_col)['Net Amount'].sum()
                age_spend.plot(kind='bar', color='#66b3ff')
                plt.title('Spending by Age Group')
                plt.xticks(rotation=45)
                save_plot("age_spending.png")
                pdf.add_image("age_spending.png", "Spending by Age Group")
                #plt.show()
    except Exception as e:
        log(f"Error in gender/age analysis: {e}")

def product_category_analysis(data, age_col, pdf):
    product_col = check_product_column(data)
    if product_col:
        try:
            log(f"Performing {product_col} analysis...")
            if 'Net Amount' in data.columns:
                plt.figure(figsize=(12, 6))
                product_spend = data.groupby(product_col)['Net Amount'].sum().sort_values(ascending=False)
                product_spend.plot(kind='bar', color='#66b3ff')
                plt.title(f'{product_col} by Total Net Amount')
                plt.xlabel(product_col)
                plt.ylabel('Total Net Amount')
                plt.xticks(rotation=45)
                save_plot("top_product_spend.png")
                pdf.add_image("top_product_spend.png", f"{product_col} by Total Net Amount")
                #plt.show()
                log(f"Top {product_col} by Net Amount:\n{product_spend}")

            if age_col:
                plt.figure(figsize=(12, 6))
                top_products = data[product_col].value_counts().index
                filtered_data = data[data[product_col].isin(top_products)]
                sns.countplot(x=age_col, hue=product_col, data=filtered_data, palette='Set2')
                plt.title(f'{product_col}s by Age Group')
                plt.xticks(rotation=45)
                plt.legend(title=product_col, bbox_to_anchor=(1.05, 1), loc='upper left')
                save_plot("age_product_category.png")
                pdf.add_image("age_product_category.png", f"{product_col}s by Age Group")
                #plt.show()
        except Exception as e:
            log(f"Error in {product_col} analysis: {e}")

def behavioral_segmentation(data, pdf):
    try:
        log("Performing behavioral segmentation...")
        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], errors='coerce')
        if data['InvoiceDate'].isna().all():
            raise ValueError("Invalid or missing InvoiceDate values.")
        data['Hour'] = data['InvoiceDate'].dt.hour
        data['DayType'] = data['InvoiceDate'].dt.weekday.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
        data['Month'] = data['InvoiceDate'].dt.to_period('M')

        plt.figure(figsize=(10, 5))
        sns.countplot(x='Hour', data=data, palette='Set2')
        plt.title('Purchase Time Analysis (Hour of Day)')
        save_plot("purchase_time_hour.png")
        pdf.add_image("purchase_time_hour.png", "Purchase Time Analysis (Hour of Day)")
        #plt.show()

        plt.figure(figsize=(6, 5))
        sns.countplot(x='DayType', data=data, palette='Set2')
        plt.title('Day Type Analysis (Weekday vs Weekend)')
        save_plot("purchase_day_type.png")
        pdf.add_image("purchase_day_type.png", "Day Type Analysis (Weekday vs Weekend)")
        #plt.show()

        plt.figure(figsize=(12, 5))
        sns.countplot(x='Month', data=data, palette='Set2')
        plt.title('Retail Trend by Month')
        plt.xticks(rotation=45)
        save_plot("purchase_trend_month.png")
        pdf.add_image("purchase_trend_month.png", "Retail Trend by Month")
        #plt.show()
    except Exception as e:
        log(f"Error in behavioral segmentation: {e}")

def preprocess_rfm(rfm, data, gender_col, age_col):
    try:
        log("Preprocessing RFM data...")
        features = rfm[['Recency', 'Frequency', 'Monetary']].copy()

        if gender_col:
            gender_data = data[['CustomerID', gender_col]].groupby('CustomerID').agg(lambda x: x.mode().iloc[0] if not x.empty else np.nan).reset_index()
            gender_data = gender_data.drop_duplicates(subset='CustomerID').set_index('CustomerID')
            rfm = rfm.join(gender_data, on='CustomerID')
            if gender_col in rfm.columns:
                features = pd.concat([features, pd.get_dummies(rfm[gender_col], prefix='Gender', dummy_na=False)], axis=1)

        if age_col:
            age_data = data[['CustomerID', age_col]].groupby('CustomerID').agg(lambda x: x.mode().iloc[0] if not x.empty else np.nan).reset_index()
            age_data = age_data.drop_duplicates(subset='CustomerID').set_index('CustomerID')
            rfm = rfm.join(age_data, on='CustomerID')
            if age_col in rfm.columns:
                features = pd.concat([features, pd.get_dummies(rfm[age_col], prefix='AgeGroup', dummy_na=False)], axis=1)

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        return scaled_features, features.columns.tolist(), rfm
    except Exception as e:
        log(f"Error in preprocessing RFM: {e}")
        return None, None, rfm

def select_features(data):
    try:
        log("Selecting features for clustering...")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            log("No numeric columns found for clustering.")
            return None, None

        corr_matrix = data[numeric_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
        selected_numeric = [col for col in numeric_cols if col not in to_drop]

        categorical_cols = data.select_dtypes(include=['object']).columns
        categorical_features = list(categorical_cols)

        features = data[selected_numeric]
        for col in categorical_features:
            dummies = pd.get_dummies(data[col], prefix=col, dummy_na=False)
            features = pd.concat([features, dummies], axis=1)

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features.dropna())
        return scaled_features, features.columns.tolist(), features
    except Exception as e:
        log(f"Error in feature selection: {e}")
        return None, None, None

def visualize_features(data, clusters, features, pdf, is_rfm=False, data_original=None):
    try:
        log("Generating visualizations...")
        df = pd.DataFrame(data, columns=features)
        df['Cluster'] = clusters
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = [col for col in features if col not in numeric_cols]

        # Add Churn_Risk to df if available in data_original
        if data_original is not None and 'Churn_Risk' in data_original.columns:
            df['Churn_Risk'] = data_original['Churn_Risk']
        else:
            df['Churn_Risk'] = 'Low'  # Default if not available

        # Sample data for pair plot if dataset is large
        sample_size = min(1000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df

        # Pair plot for numeric features
        if len(numeric_cols) > 1:
            log("Generating pair plot...")
            sns.pairplot(df_sample, hue='Cluster', vars=numeric_cols, palette='Set2')
            plt.suptitle('Pair Plot of Numeric Features by Cluster', y=1.02)
            save_plot("pair_plot.png")
            pdf.add_image("pair_plot.png", "Pair Plot of Numeric Features by Cluster")
            #plt.show()

        # Scatter plots for each pair of numeric features
        if len(numeric_cols) > 1:
            log("Generating scatter plots for numeric feature pairs...")
            from itertools import combinations
            for col1, col2 in combinations(numeric_cols, 2):
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=df_sample, x=col1, y=col2, hue='Cluster', palette='Set2')
                plt.title(f'{col1} vs {col2} by Cluster')
                save_plot(f"scatter_{col1}_vs_{col2}.png")
                pdf.add_image(f"scatter_{col1}_vs_{col2}.png", f"{col1} vs {col2} by Cluster")
                #plt.show()

        # Correlation heatmap for numeric features
        if len(numeric_cols) > 1:
            log("Generating correlation heatmap...")
            plt.figure(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Correlation Heatmap of Numeric Features')
            save_plot("correlation_heatmap.png")
            pdf.add_image("correlation_heatmap.png", "Correlation Heatmap of Numeric Features")
            #plt.show()

        # Distribution plots for numeric features by cluster
        for col in numeric_cols:
            log(f"Generating distribution plot for {col}...")
            plt.figure(figsize=(10, 5))
            for cluster in sorted(df['Cluster'].unique()):
                sns.histplot(df[df['Cluster'] == cluster][col], label=f'Cluster {cluster}', kde=True, element='step')
            plt.title(f'Distribution of {col} by Cluster')
            plt.legend()
            save_plot(f"dist_{col}.png")
            pdf.add_image(f"dist_{col}.png", f"Distribution of {col} by Cluster")
            #plt.show()

        # Violin plots for numeric features by cluster
        for col in numeric_cols:
            log(f"Generating violin plot for {col}...")
            plt.figure(figsize=(10, 5))
            sns.violinplot(x='Cluster', y=col, data=df, palette='Set2')
            plt.title(f'{col} Distribution by Cluster (Violin Plot)')
            plt.xticks(rotation=45)
            save_plot(f"violin_{col}.png")
            pdf.add_image(f"violin_{col}.png", f"{col} Distribution by Cluster (Violin Plot)")
            #plt.show()

        # Bar plots for categorical features by cluster
        for col in categorical_cols:
            log(f"Generating bar plot for {col}...")
            plt.figure(figsize=(10, 5))
            sns.countplot(data=df, x=col, hue='Cluster', palette='Set2')
            plt.title(f'{col} Distribution by Cluster')
            plt.xticks(rotation=45)
            plt.legend(title='Cluster')
            save_plot(f"cat_{col}.png")
            pdf.add_image(f"cat_{col}.png", f"{col} Distribution by Cluster")
            #plt.show()

        # Churn risk by cluster
        if 'Churn_Risk' in df.columns:
            log("Generating churn risk by cluster plot...")
            plt.figure(figsize=(10, 5))
            sns.countplot(data=df, x='Cluster', hue='Churn_Risk', palette='Set2')
            plt.title('Churn Risk Distribution by Cluster')
            plt.xticks(rotation=45)
            plt.legend(title='Churn Risk')
            save_plot("churn_risk_by_cluster.png")
            pdf.add_image("churn_risk_by_cluster.png", "Churn Risk Distribution by Cluster")
            #plt.show()

        # Cluster size distribution
        log("Generating cluster size distribution...")
        plt.figure(figsize=(8, 5))
        sns.countplot(x=df['Cluster'], palette='Set2')
        plt.title('Cluster Size Distribution')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Customers')
        save_plot("cluster_sizes.png")
        pdf.add_image("cluster_sizes.png", "Cluster Size Distribution")
        #plt.show()

        # Boxplots for numeric features by cluster
        if len(numeric_cols) > 0:
            log("Generating boxplots...")
            n_cols = min(len(numeric_cols), 3)
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            plt.figure(figsize=(15, 5 * n_rows))
            for i, col in enumerate(numeric_cols):
                plt.subplot(n_rows, n_cols, i + 1)
                sns.boxplot(x='Cluster', y=col, data=df, palette='Set2')
                plt.title(f'{col} by Cluster')
                plt.xticks(rotation=45)
            plt.tight_layout()
            save_plot("cluster_boxplot.png")
            pdf.add_image("cluster_boxplot.png", "Cluster Profiles (Boxplots)")
            #plt.show()

        # 2D PCA scatter plot
        if len(features) > 1:
            log("Generating PCA scatter plot...")
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(data)
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=clusters, cmap='Set2')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.title('Customer Segmentation (2D PCA View)')
            plt.colorbar(scatter, label='Cluster')
            save_plot("cluster_pca.png")
            pdf.add_image("cluster_pca.png", "Customer Segmentation (2D PCA View)")
            #plt.show()
    except Exception as e:
        log(f"Error in visualizing features: {e}")

def determine_optimal_clusters(data, min_clusters=2, pdf=None):
    try:
        log("Determining optimal number of clusters...")
        n_samples = len(data)
        max_clusters = 10  # Fixed maximum clusters as requested
        if n_samples < min_clusters:
            log(f"Dataset too small ({n_samples} samples) for clustering. Defaulting to {min_clusters} clusters.")
            return min_clusters
        if n_samples < max_clusters:
            max_clusters = n_samples // 2  # Ensure max_clusters doesn't exceed half the sample size

        silhouette_scores = []
        inertia = []
        tested_k = range(min_clusters, max_clusters + 1)
        for k in tested_k:
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=100)
            kmeans.fit(data)
            silhouette_scores.append(silhouette_score(data, kmeans.labels_))
            inertia.append(kmeans.inertia_)

        best_k = np.argmax(silhouette_scores) + min_clusters

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(tested_k, silhouette_scores, marker='o', color='#66b3ff')
        plt.title('Silhouette Scores for Different k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(tested_k, inertia, marker='o', color='#ff9999')
        plt.title('Elbow Method (Inertia)')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.grid(True)
        plt.tight_layout()
        save_plot("cluster_evaluation.png")
        pdf.add_image("cluster_evaluation.png", "Cluster Evaluation")
        #plt.show()

        log(f"Silhouette Scores: {silhouette_scores}")
        log(f"Inertia: {inertia}")
        log(f"Optimal number of clusters based on silhouette score: {best_k}")
        return best_k
    except Exception as e:
        log(f"Error in determining optimal clusters: {e}")
        return min_clusters

def perform_clustering(data, n_clusters):
    try:
        log(f"Performing clustering with {n_clusters} clusters...")
        return MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=100).fit_predict(data)
    except Exception as e:
        log(f"Error in performing clustering: {e}")
        return np.zeros(len(data), dtype=int)

def marketing_recommendations(rfm, clusters, gender_col, age_col, pdf):
    try:
        log("Generating marketing recommendations...")
        rfm['Cluster'] = clusters
        recommendations_text = "Marketing Recommendations:\n"
        for cluster in sorted(rfm['Cluster'].unique()):
            segment = rfm[rfm['Cluster'] == cluster]
            total_customers = len(segment)
            best_customers_pct = (segment['RFM_Tile'] == '⭐ Best Customers').mean() * 100
            high_churn_pct = (segment['Churn_Risk'] == '⚠️ High').mean() * 100

            recommendations = [f"Cluster {cluster} ({total_customers} customers):\n"]
            if best_customers_pct > 50:
                recommendations.append(f"- Upsell premium products ({best_customers_pct:.1f}% are Best Customers)\n")
            elif high_churn_pct > 50:
                recommendations.append(f"- Win-back campaigns ({high_churn_pct:.1f}% at High Churn Risk)\n")
            else:
                recommendations.append("- Engage with loyalty rewards\n")

            if gender_col and gender_col in segment.columns:
                gender_dist = segment[gender_col].value_counts(normalize=True) * 100
                if not gender_dist.empty:
                    dominant_gender = gender_dist.idxmax()
                    recommendations.append(f"- Target {dominant_gender} (dominant gender: {gender_dist[dominant_gender]:.1f}%)\n")

            if age_col and age_col in segment.columns:
                age_dist = segment[age_col].value_counts(normalize=True) * 100
                if not age_dist.empty:
                    dominant_age = age_dist.idxmax()
                    recommendations.append(f"- Focus on {dominant_age} age group ({age_dist[dominant_age]:.1f}%)\n")

            recommendations_text += "".join(recommendations) + "\n"
        pdf.add_title("Marketing Recommendations")
        pdf.add_text(recommendations_text)
    except Exception as e:
        log(f"Error in marketing recommendations: {e}")

def marketing_recommendations_non_rfm(data, clusters, features, pdf):
    try:
        log("Generating non-RFM marketing recommendations...")
        data['Cluster'] = clusters
        recommendations = "Marketing Recommendations:\n"
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns

        for cluster in sorted(data['Cluster'].unique()):
            segment = data[data['Cluster'] == cluster]
            rec = f"Cluster {cluster} ({len(segment)} customers):\n"

            for col in numeric_cols:
                mean_val = segment[col].mean()
                overall_mean = data[col].mean()
                if mean_val > overall_mean * 1.2:
                    rec += f"- High {col} (avg: {mean_val:.2f} vs overall: {overall_mean:.2f})\n"
                elif mean_val < overall_mean * 0.8:
                    rec += f"- Low {col} (avg: {mean_val:.2f} vs overall: {overall_mean:.2f})\n"

            for col in categorical_cols:
                if col in segment.columns:
                    dist = segment[col].value_counts(normalize=True) * 100
                    if not dist.empty:
                        dominant = dist.idxmax()
                        rec += f"- Dominant {col}: {dominant} ({dist[dominant]:.1f}%)\n"

            if 'Churn_Risk' in data.columns:
                high_churn_pct = (segment['Churn_Risk'] == 'High').mean() * 100
                rec += f"- Churn Risk: {high_churn_pct:.1f}% High Risk\n"
                if high_churn_pct > 50:
                    rec += "- Implement win-back campaigns for high-risk customers\n"
                else:
                    rec += "- Maintain engagement with loyalty programs\n"

            recommendations += rec + "\n"
        pdf.add_title("Marketing Recommendations (Non-RFM)")
        pdf.add_text(recommendations)
    except Exception as e:
        log(f"Error in non-RFM recommendations: {e}")

def check_rfm_columns(data):
    required_cols = ['CustomerID', 'InvoiceDate', 'InvoiceNo']
    monetary_cols = ['Net Amount'] if 'Net Amount' in data.columns else ['Quantity', 'UnitPrice']
    return all(col in data.columns for col in required_cols) and any(col in data.columns for col in monetary_cols)

def normalize_column_names(data):
    column_mappings = {
    'CustomerID': [
        'customerid', 'custid', 'cust_id', 'userid', 'user_id',
        'clientid', 'client_id', 'memberid', 'member_id', 'buyer_id'
    ],
    'InvoiceNo': [
        'invoiceno', 'invoice_no', 'transaction_id', 'transactionid', 'tid',
        'orderid', 'order_id', 'billno', 'bill_no', 'receiptno', 'receipt_id'
    ],
    'InvoiceDate': [
        'invoicedate', 'invoice_date', 'transaction_date', 'date', 'timestamp',
        'transactiontime', 'order_date', 'datetime', 'purchase_date'
    ],
    'Net Amount': [
        'netamount', 'net_amount', 'amount', 'total_amount', 'total',
        'price_total', 'order_total', 'totalprice', 'amount_paid', 'net_price'
    ],
    'UnitPrice': [
        'unitprice', 'unit_price', 'price', 'rate', 'unit_cost', 'costperitem',
        'item_price', 'itemcost', 'selling_price', 'product_price'
    ],
    'Quantity': [
        'quantity', 'qty', 'count', 'number_of_items', 'item_count',
        'numberofitemspurchased', 'units', 'product_quantity', 'qty_sold'
    ]
}


    lowercase_columns = {col.lower(): col for col in data.columns}

    for standard_col, aliases in column_mappings.items():
        for alias in aliases:
            if alias.lower() in lowercase_columns:
                original_col = lowercase_columns[alias.lower()]
                data = data.rename(columns={original_col: standard_col})
                break  # use the first matched alias
    return data

def run_analysis_from_file(file_path, output_pdf="report.pdf"):
    from analyse import AnalysisPDF
    data = pd.read_csv(file_path)

    # Normalize & check
    data = normalize_column_names(data)
    gender_col, age_col = check_gender_age_columns(data)

    pdf = AnalysisPDF()
    pdf.add_page()
    pdf.add_title("Introduction")
    pdf.add_text("This report presents the results of the customer segmentation analysis...")

    if check_rfm_columns(data):
        rfm = compute_rfm(data, pdf)
        rfm = assign_rfm_tiles(rfm)
        rfm = churn_prediction(rfm)
        scaled_data, feature_names, rfm = preprocess_rfm(rfm, data, gender_col, age_col)
        optimal_k = determine_optimal_clusters(scaled_data, pdf=pdf)
        clusters = perform_clustering(scaled_data, optimal_k)
        visualize_features(scaled_data, clusters, feature_names, pdf, is_rfm=True, data_original=rfm)
        marketing_recommendations(rfm, clusters, gender_col, age_col, pdf)
    else:
        scaled_data, feature_names, features_df = select_features(data)
        data = churn_prediction_non_rfm(data, numeric_cols=features_df.select_dtypes(include=[np.number]).columns)
        optimal_k = determine_optimal_clusters(scaled_data, pdf=pdf)
        clusters = perform_clustering(scaled_data, optimal_k)
        visualize_features(scaled_data, clusters, feature_names, pdf, is_rfm=False, data_original=data)
        marketing_recommendations_non_rfm(data, clusters, feature_names, pdf)

    gender_age_analysis(data, gender_col, age_col, pdf)
    product_category_analysis(data, age_col, pdf)
    behavioral_segmentation(data, pdf)
    pdf.output(name=output_pdf, dest='F')
    print(f"[Flask] ✅ Report saved as {output_pdf}")

    # Clean up all generated .png files
    for file in os.listdir():
        if file.endswith(".png"):
            try:
                os.remove(file)
            except Exception as e:
                print(f"Failed to delete {file}: {e}")

    return output_pdf



if __name__ == "__main__":
    run_analysis_from_file()
