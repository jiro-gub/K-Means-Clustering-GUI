import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page Configuration
st.set_page_config(
    page_title='K-Means Clustering GUI',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Main Title and Description
st.title('📊 Interactive K-Means Clustering')
st.markdown("""
This application allows you to perform K-means clustering on your dataset:
1. Upload a CSV file using the sidebar
2. Select numeric features for clustering
3. Choose the number of clusters (k)
4. View interactive visualizations and metrics

The app includes:
- Automatic data preprocessing
- Elbow point detection
- Multiple clustering quality metrics
- Interactive visualizations
""")

# Sidebar
st.sidebar.header('📈 Data Upload and Parameters')

# File Upload
uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file",
    type=['csv'],
    help="Upload a CSV file containing numeric features for clustering"
)

if uploaded_file is not None:
    try:
        # Load and display data info
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f'Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns')
        
        # Data Overview
        with st.expander("Dataset Overview"):
            st.write("First few rows of the dataset:")
            st.dataframe(df.head(), use_container_width=True)
            
            st.write("Dataset Information:")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Missing Values:")
                st.dataframe(df.isnull().sum().to_frame('Missing Count'), use_container_width=True)
            with col2:
                st.write("Data Types:")
                st.dataframe(df.dtypes.to_frame('Data Types'), use_container_width=True)
        
        # Feature Selection
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_columns) < 2:
            st.error('The dataset must contain at least 2 numeric columns!')
            st.stop()
            
        selected_features = st.sidebar.multiselect(
            'Select features for clustering:',
            options=numeric_columns,
            default=numeric_columns[:2] if len(numeric_columns) >= 2 else [],
            help="Choose at least 2 numeric features for clustering"
        )
        
        # Preprocessing Options
        with st.sidebar.expander("Preprocessing Options"):
            scale_data = st.checkbox('Scale features (recommended)', value=True,
                                   help="Standardize features by removing the mean and scaling to unit variance")
            remove_outliers = st.checkbox('Remove outliers', value=False,
                                        help="Remove samples that are more than 3 standard deviations from the mean")
        
        # Clustering Parameters
        k_clusters = st.sidebar.number_input(
            'Number of clusters (k)',
            min_value=2,
            max_value=10,
            value=3,
            help="Choose the number of clusters to create"
        )
        
        random_state = st.sidebar.slider(
            'Random seed',
            min_value=0,
            max_value=100,
            value=42,
            help="Set random seed for reproducibility"
        )
        
        compute_elbow = st.sidebar.checkbox(
            'Compute Elbow method',
            value=True,
            help="Calculate and plot the elbow curve with automatic detection"
        )
        
        compute_metrics = st.sidebar.checkbox(
            'Compute clustering metrics',
            value=True,
            help="Calculate silhouette and Calinski-Harabasz scores"
        )
        
        if len(selected_features) < 2:
            st.warning('Please select at least 2 features for clustering.')
            st.stop()
            
        # Clustering Logic
        with st.spinner('Running K-Means clustering...'):
            # Prepare data
            X = df[selected_features].dropna()
            
            # Preprocessing
            if scale_data:
                scaler = StandardScaler()
                X_processed = scaler.fit_transform(X)
                X_processed = pd.DataFrame(X_processed, columns=X.columns)
            else:
                X_processed = X.copy()
            
            if remove_outliers:
                z_scores = np.abs((X_processed - X_processed.mean()) / X_processed.std())
                X_processed = X_processed[(z_scores < 3).all(axis=1)]
            
            # Fit K-means
            kmeans = KMeans(
                n_clusters=k_clusters,
                random_state=random_state,
                n_init=10
            )
            cluster_labels = kmeans.fit_predict(X_processed)
            
            # Add cluster labels to dataframe
            results_df = X_processed.copy()
            results_df['Cluster'] = cluster_labels
            
            # Main Panel
            st.header('🎯 Clustering Results')
            
            # Display results dataframe
            st.subheader('Clustered Data')
            st.dataframe(
                results_df,
                use_container_width=True
            )
            
            # Visualizations
            viz_col1, viz_col2 = st.columns(2)
            
            # Scatter plot
            with viz_col1:
                st.subheader('Cluster Visualization')
                if len(selected_features) >= 2:
                    # Create scatter plot with samples
                    fig = px.scatter(
                        results_df,
                        x=selected_features[0],
                        y=selected_features[1],
                        color='Cluster',
                        title=f'K-means Clustering Results: {selected_features[0]} vs {selected_features[1]}',
                        hover_data=selected_features,
                        color_continuous_scale='viridis'
                    )
                    
                    # Add cluster centers
                    centers = pd.DataFrame(
                        kmeans.cluster_centers_,
                        columns=selected_features
                    )
                    
                    # Add centroids to the plot
                    fig.add_trace(
                        go.Scatter(
                            x=centers[selected_features[0]],
                            y=centers[selected_features[1]],
                            mode='markers',
                            marker=dict(
                                color='red',
                                size=15,
                                symbol='x',
                                line=dict(color='black', width=2)
                            ),
                            name='Centroids',
                            hovertext=[f'Centroid {i}' for i in range(len(centers))]
                        )
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title=selected_features[0],
                        yaxis_title=selected_features[1],
                        showlegend=True,
                        legend_title_text='Clusters',
                        height=600
                    )
                    
                    # If more than 2 features were selected, add a note
                    if len(selected_features) > 2:
                        st.info("Note: The visualization shows the first two selected features. Other features are used in clustering but not shown in this 2D plot.")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add feature correlation info
                    if len(selected_features) > 2:
                        with st.expander("Feature Correlations"):
                            corr = results_df[selected_features].corr()
                            fig_corr = px.imshow(
                                corr,
                                title='Feature Correlation Matrix',
                                color_continuous_scale='RdBu_r',
                                aspect='auto'
                            )
                            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Cluster Distribution
            with viz_col2:
                st.subheader('Cluster Distribution')
                cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
                fig = px.bar(
                    data_frame=pd.DataFrame({
                        'Cluster': cluster_counts.index,
                        'Count': cluster_counts.values
                    }),
                    x='Cluster',
                    y='Count',
                    title='Number of Samples per Cluster',
                    labels={'Count': 'Number of Samples', 'Cluster': 'Cluster Label'}
                )
                fig.update_traces(marker_color='rgb(55, 83, 109)')
                st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            if compute_metrics:
                st.subheader('Clustering Quality Metrics')
                metrics_col1, metrics_col2 = st.columns(2)
                
                # Silhouette Score
                with metrics_col1:
                    silhouette_avg = silhouette_score(X_processed, cluster_labels)
                    st.metric(
                        'Silhouette Score',
                        f'{silhouette_avg:.3f}',
                        help='Ranges from -1 to 1, higher is better'
                    )
                
                # Calinski-Harabasz Score
                with metrics_col2:
                    ch_score = calinski_harabasz_score(X_processed, cluster_labels)
                    st.metric(
                        'Calinski-Harabasz Score',
                        f'{ch_score:.1f}',
                        help='Higher values indicate better defined clusters'
                    )
            
            # Elbow Method with Automatic Detection
            if compute_elbow:
                st.subheader('Elbow Method Analysis')
                
                inertias = []
                k_values = range(1, 11)
                
                for k in k_values:
                    kmeans_temp = KMeans(
                        n_clusters=k,
                        random_state=random_state,
                        n_init=10
                    )
                    kmeans_temp.fit(X_processed)
                    inertias.append(kmeans_temp.inertia_)
                
                # Find elbow point
                kn = KneeLocator(
                    list(k_values),
                    inertias,
                    curve='convex',
                    direction='decreasing'
                )
                
                # Plot with detected elbow point
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(k_values),
                    y=inertias,
                    mode='lines+markers',
                    name='Inertia'
                ))
                
                if kn.elbow is not None:
                    fig.add_trace(go.Scatter(
                        x=[kn.elbow],
                        y=[kn.elbow_y],
                        mode='markers',
                        marker=dict(size=10, color='red'),
                        name=f'Elbow Point (k={kn.elbow})'
                    ))
                
                fig.update_layout(
                    title='Elbow Method for Optimal k',
                    xaxis_title='Number of Clusters (k)',
                    yaxis_title='Inertia',
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
                if kn.elbow is not None:
                    st.info(f'🎯 The optimal number of clusters detected is: {kn.elbow}')
            
            # Algorithm Explanation
            with st.expander('Show Algorithm Details'):
                st.markdown("""
                ### K-means Clustering Algorithm Steps
                
                1. **Initialization**
                   - Randomly initialize k centroids in the feature space
                   - Using k-means++ for smarter initialization (default in scikit-learn)
                
                2. **Assignment Step**
                   - Assign each data point to nearest centroid using Euclidean distance
                   ```python
                   distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
                   labels = distances.argmin(axis=0)
                   ```
                
                3. **Update Step**
                   - Compute new centroids as mean of assigned points
                   ```python
                   new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(n_clusters)])
                   ```
                
                4. **Convergence**
                   - Repeat steps 2-3 until centroids stabilize or max iterations reached
                
                ### Preprocessing Steps Applied
                - Feature scaling (StandardScaler)
                - Outlier removal (optional)
                
                ### Evaluation Metrics
                - **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters
                - **Calinski-Harabasz Score**: Ratio of between-cluster dispersion and within-cluster dispersion
                - **Inertia**: Sum of squared distances to closest centroid (used in elbow method)
                """)
    
    except Exception as e:
        st.error(f'Error loading or processing the file: {str(e)}')
        st.stop()

else:
    st.sidebar.info('Please upload a CSV file to begin.')
    st.info('👈 Start by uploading your data using the sidebar.')
    st.stop()

# Footer
st.markdown('---')
st.caption('Built with Streamlit • K-means Clustering GUI') 