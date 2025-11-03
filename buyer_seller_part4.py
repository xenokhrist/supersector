"""
Part 4: Analytics, Visualization, and Export Tools

Provides:
- Performance analytics
- Tag importance analysis
- Network visualization (optional)
- Export utilities
- Quality metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
import io
from datetime import datetime

try:
    from pyvis.network import Network
    import networkx as nx
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

# =============================================================================
# SIMILARITY MATRIX ANALYTICS
# =============================================================================

def analyze_similarity_matrix(
    similarity_matrix: np.ndarray,
    threshold: float = 0.65
) -> Dict[str, any]:
    """
    Analyze similarity matrix statistics
    
    Returns comprehensive metrics
    """
    
    # Flatten for statistics
    flat_scores = similarity_matrix.flatten()
    
    # Basic statistics
    stats = {
        'total_combinations': similarity_matrix.size,
        'n_buyers': similarity_matrix.shape[0],
        'n_sellers': similarity_matrix.shape[1],
        'mean': float(np.mean(flat_scores)),
        'median': float(np.median(flat_scores)),
        'std': float(np.std(flat_scores)),
        'min': float(np.min(flat_scores)),
        'max': float(np.max(flat_scores)),
        'q25': float(np.percentile(flat_scores, 25)),
        'q75': float(np.percentile(flat_scores, 75)),
    }
    
    # Threshold-based metrics
    above_threshold = flat_scores > threshold
    stats['above_threshold_count'] = int(np.sum(above_threshold))
    stats['above_threshold_pct'] = float(np.mean(above_threshold) * 100)
    
    # Quality tiers
    stats['excellent_matches'] = int(np.sum(flat_scores > 0.8))  # >80%
    stats['good_matches'] = int(np.sum((flat_scores > 0.65) & (flat_scores <= 0.8)))  # 65-80%
    stats['moderate_matches'] = int(np.sum((flat_scores > 0.5) & (flat_scores <= 0.65)))  # 50-65%
    stats['weak_matches'] = int(np.sum(flat_scores <= 0.5))  # <50%
    
    # Per-entity statistics
    buyer_max_scores = similarity_matrix.max(axis=1)
    seller_max_scores = similarity_matrix.max(axis=0)
    
    stats['buyers_with_good_match'] = int(np.sum(buyer_max_scores > threshold))
    stats['sellers_with_good_match'] = int(np.sum(seller_max_scores > threshold))
    
    stats['buyers_no_match'] = int(np.sum(buyer_max_scores <= 0.3))
    stats['sellers_no_match'] = int(np.sum(seller_max_scores <= 0.3))
    
    return stats

def display_similarity_statistics(stats: Dict[str, any]):
    """Display similarity statistics in Streamlit"""
    
    st.subheader("📊 Similarity Matrix Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Combinations", f"{stats['total_combinations']:,}")
        st.metric("Mean Similarity", f"{stats['mean']:.3f}")
    
    with col2:
        st.metric("Buyers", f"{stats['n_buyers']:,}")
        st.metric("Median Similarity", f"{stats['median']:.3f}")
    
    with col3:
        st.metric("Sellers", f"{stats['n_sellers']:,}")
        st.metric("Std Deviation", f"{stats['std']:.3f}")
    
    with col4:
        st.metric("Above Threshold", f"{stats['above_threshold_count']:,}")
        st.metric("Max Similarity", f"{stats['max']:.3f}")
    
    # Quality distribution
    st.subheader("🎯 Match Quality Distribution")
    
    quality_col1, quality_col2 = st.columns(2)
    
    with quality_col1:
        # Pie chart
        quality_data = pd.DataFrame({
            'Quality': ['Excellent (>0.8)', 'Good (0.65-0.8)', 'Moderate (0.5-0.65)', 'Weak (<0.5)'],
            'Count': [
                stats['excellent_matches'],
                stats['good_matches'],
                stats['moderate_matches'],
                stats['weak_matches']
            ]
        })
        
        fig_pie = px.pie(
            quality_data,
            values='Count',
            names='Quality',
            title='Match Quality Distribution',
            color_discrete_sequence=['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with quality_col2:
        # Bar chart
        fig_bar = px.bar(
            quality_data,
            x='Quality',
            y='Count',
            title='Match Counts by Quality Tier',
            color='Quality',
            color_discrete_sequence=['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Coverage statistics
    st.subheader("📈 Entity Coverage")
    
    cov_col1, cov_col2 = st.columns(2)
    
    with cov_col1:
        st.metric(
            "Buyers with Good Match",
            f"{stats['buyers_with_good_match']:,}",
            f"{stats['buyers_with_good_match']/stats['n_buyers']*100:.1f}%"
        )
        st.metric(
            "Buyers with No Match",
            f"{stats['buyers_no_match']:,}",
            f"{stats['buyers_no_match']/stats['n_buyers']*100:.1f}%"
        )
    
    with cov_col2:
        st.metric(
            "Sellers with Good Match",
            f"{stats['sellers_with_good_match']:,}",
            f"{stats['sellers_with_good_match']/stats['n_sellers']*100:.1f}%"
        )
        st.metric(
            "Sellers with No Match",
            f"{stats['sellers_no_match']:,}",
            f"{stats['sellers_no_match']/stats['n_sellers']*100:.1f}%"
        )

# =============================================================================
# DISTRIBUTION VISUALIZATIONS
# =============================================================================

def plot_similarity_distribution(similarity_matrix: np.ndarray, sample_size: int = 50000):
    """Plot similarity score distribution"""
    
    flat_scores = similarity_matrix.flatten()
    
    # Sample for large matrices
    if len(flat_scores) > sample_size:
        flat_scores = np.random.choice(flat_scores, sample_size, replace=False)
        st.caption(f"📊 Showing sample of {sample_size:,} scores (total: {similarity_matrix.size:,})")
    
    # Histogram
    fig = px.histogram(
        x=flat_scores,
        nbins=50,
        title="Similarity Score Distribution",
        labels={'x': 'Similarity Score', 'y': 'Frequency'},
        color_discrete_sequence=['#3498db']
    )
    
    fig.add_vline(x=0.65, line_dash="dash", line_color="red", 
                  annotation_text="Threshold (0.65)", annotation_position="top")
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def plot_top_matches_heatmap(
    similarity_matrix: np.ndarray,
    buyers_df: pd.DataFrame,
    sellers_df: pd.DataFrame,
    buyer_name_col: str,
    seller_name_col: str,
    max_display: int = 30
):
    """Plot heatmap of top matches"""
    
    n_buyers = min(len(buyers_df), max_display)
    n_sellers = min(len(sellers_df), max_display)
    
    # Get top buyers and sellers by max similarity
    buyer_max = similarity_matrix.max(axis=1)
    seller_max = similarity_matrix.max(axis=0)
    
    top_buyers = np.argsort(-buyer_max)[:n_buyers]
    top_sellers = np.argsort(-seller_max)[:n_sellers]
    
    # Extract submatrix
    sub_matrix = similarity_matrix[np.ix_(top_buyers, top_sellers)]
    
    # Create labels
    buyer_labels = [buyers_df.iloc[i][buyer_name_col][:20] for i in top_buyers]
    seller_labels = [sellers_df.iloc[i][seller_name_col][:20] for i in top_sellers]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=sub_matrix,
        x=seller_labels,
        y=buyer_labels,
        colorscale='Viridis',
        colorbar=dict(title="Similarity"),
        hoverongaps=False,
        hovertemplate='Buyer: %{y}<br>Seller: %{x}<br>Similarity: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Top {n_buyers}×{n_sellers} Matches Heatmap",
        xaxis_title="Sellers",
        yaxis_title="Buyers",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TOP MATCHES TABLE
# =============================================================================

def get_top_global_matches(
    similarity_matrix: np.ndarray,
    buyers_df: pd.DataFrame,
    sellers_df: pd.DataFrame,
    buyer_name_col: str,
    seller_name_col: str,
    top_n: int = 20
) -> pd.DataFrame:
    """Get top N global matches"""
    
    # Find top matches
    flat_matrix = similarity_matrix.flatten()
    top_indices = np.argpartition(-flat_matrix, top_n-1)[:top_n]
    top_indices = top_indices[np.argsort(-flat_matrix[top_indices])]
    
    # Convert to 2D indices
    top_2d_indices = np.unravel_index(top_indices, similarity_matrix.shape)
    
    # Create results
    results = []
    for rank, (buyer_idx, seller_idx) in enumerate(zip(top_2d_indices[0], top_2d_indices[1]), 1):
        buyer_name = buyers_df.iloc[buyer_idx][buyer_name_col]
        seller_name = sellers_df.iloc[seller_idx][seller_name_col]
        score = similarity_matrix[buyer_idx, seller_idx]
        
        result = {
            'Rank': rank,
            'Buyer': buyer_name[:40],
            'Seller': seller_name[:40],
            'Score': f"{score:.3f}"
        }
        
        # Add Stand_no if available
        if 'Stand_no' in sellers_df.columns:
            result['Stand_no'] = sellers_df.iloc[seller_idx]['Stand_no']
        
        results.append(result)
    
    return pd.DataFrame(results)

# =============================================================================
# TAG IMPORTANCE ANALYSIS
# =============================================================================

def analyze_tag_importance_by_level(
    hierarchical_tags_buyers: Dict[str, List[List[str]]],
    hierarchical_tags_sellers: Dict[str, List[List[str]]]
) -> pd.DataFrame:
    """
    Analyze tag importance across hierarchy levels
    
    Returns DataFrame with tag statistics by level
    """
    
    all_stats = []
    
    for level in ['T3', 'T1', 'T2', 'main']:
        buyer_tags = hierarchical_tags_buyers[level]
        seller_tags = hierarchical_tags_sellers[level]
        
        # Combine all tags
        all_tags = []
        for tags in buyer_tags + seller_tags:
            all_tags.extend(tags)
        
        # Count frequency
        tag_freq = {}
        for tag in all_tags:
            tag_freq[tag] = tag_freq.get(tag, 0) + 1
        
        # Get top tags for this level
        top_tags = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        
        for tag, freq in top_tags:
            all_stats.append({
                'Level': level,
                'Tag': tag,
                'Frequency': freq,
                'Weight': {'T3': 0.40, 'T1': 0.30, 'T2': 0.20, 'main': 0.10}[level],
                'Impact_Score': freq * {'T3': 0.40, 'T1': 0.30, 'T2': 0.20, 'main': 0.10}[level]
            })
    
    return pd.DataFrame(all_stats)

def plot_tag_importance(tag_importance_df: pd.DataFrame):
    """Plot tag importance analysis"""
    
    st.subheader("🏷️ Tag Importance Analysis")
    
    # Top tags by level
    fig = px.bar(
        tag_importance_df.head(30),
        x='Impact_Score',
        y='Tag',
        color='Level',
        title='Top 30 Most Important Tags (by Impact Score)',
        labels={'Impact_Score': 'Impact Score', 'Tag': 'Tag'},
        color_discrete_map={'T3': '#2ecc71', 'T1': '#3498db', 'T2': '#f39c12', 'main': '#95a5a6'},
        orientation='h'
    )
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution by level
    level_summary = tag_importance_df.groupby('Level').agg({
        'Frequency': 'sum',
        'Impact_Score': 'sum'
    }).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_freq = px.bar(
            level_summary,
            x='Level',
            y='Frequency',
            title='Tag Frequency by Level',
            color='Level',
            color_discrete_map={'T3': '#2ecc71', 'T1': '#3498db', 'T2': '#f39c12', 'main': '#95a5a6'}
        )
        st.plotly_chart(fig_freq, use_container_width=True)
    
    with col2:
        fig_impact = px.bar(
            level_summary,
            x='Level',
            y='Impact_Score',
            title='Total Impact Score by Level',
            color='Level',
            color_discrete_map={'T3': '#2ecc71', 'T1': '#3498db', 'T2': '#f39c12', 'main': '#95a5a6'}
        )
        st.plotly_chart(fig_impact, use_container_width=True)

# =============================================================================
# NETWORK VISUALIZATION (OPTIONAL)
# =============================================================================

def create_network_viz(
    similarity_matrix: np.ndarray,
    buyers_df: pd.DataFrame,
    sellers_df: pd.DataFrame,
    buyer_name_col: str,
    seller_name_col: str,
    threshold: float = 0.7,
    max_nodes: int = 100,
    max_edges: int = 200
) -> Optional[Network]:
    """Create network visualization (limited for performance)"""
    
    if not PYVIS_AVAILABLE:
        st.warning("⚠️ Pyvis not installed. Network visualization disabled.")
        return None
    
    try:
        # Get strong matches above threshold
        buyer_idx, seller_idx = np.where(similarity_matrix > threshold)
        scores = similarity_matrix[buyer_idx, seller_idx]
        
        # Limit for performance
        if len(buyer_idx) > max_edges:
            top_indices = np.argsort(-scores)[:max_edges]
            buyer_idx = buyer_idx[top_indices]
            seller_idx = seller_idx[top_indices]
            scores = scores[top_indices]
        
        # Get unique nodes
        unique_buyers = np.unique(buyer_idx)[:max_nodes//2]
        unique_sellers = np.unique(seller_idx)[:max_nodes//2]
        
        # Create network
        net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
        net.barnes_hut()
        
        # Add buyer nodes (blue)
        for b_idx in unique_buyers:
            buyer_name = buyers_df.iloc[b_idx][buyer_name_col]
            net.add_node(
                f"B{b_idx}",
                label=buyer_name[:15] + "..." if len(buyer_name) > 15 else buyer_name,
                color='#3498db',
                title=f"Buyer: {buyer_name}",
                size=30
            )
        
        # Add seller nodes (green)
        for s_idx in unique_sellers:
            seller_name = sellers_df.iloc[s_idx][seller_name_col]
            net.add_node(
                f"S{s_idx}",
                label=seller_name[:15] + "..." if len(seller_name) > 15 else seller_name,
                color='#2ecc71',
                title=f"Seller: {seller_name}",
                size=25
            )
        
        # Add edges
        for b_idx, s_idx, score in zip(buyer_idx, seller_idx, scores):
            if b_idx in unique_buyers and s_idx in unique_sellers:
                net.add_edge(
                    f"B{b_idx}",
                    f"S{s_idx}",
                    value=float(score),
                    title=f"Similarity: {score:.3f}",
                    color=f"rgba(255,165,0,{score})"
                )
        
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100},
            "barnesHut": {
              "gravitationalConstant": -80000,
              "centralGravity": 0.3,
              "springLength": 95
            }
          }
        }
        """)
        
        return net
        
    except Exception as e:
        st.error(f"Network visualization failed: {str(e)}")
        return None

# =============================================================================
# EXPORT UTILITIES
# =============================================================================

def export_performance_report(
    perf_metrics: Dict[str, float],
    stats: Dict[str, any],
    n_buyers: int,
    n_sellers: int
) -> str:
    """Generate performance report text"""
    
    report = f"""
# Buyer-Seller Matching Performance Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Information
- Buyers: {n_buyers:,}
- Sellers: {n_sellers:,}
- Total combinations: {n_buyers * n_sellers:,}
- Matches above threshold (0.65): {stats['above_threshold_count']:,} ({stats['above_threshold_pct']:.2f}%)

## Performance Metrics
"""
    
    for metric_name, metric_value in perf_metrics.items():
        if metric_value < 1:
            report += f"- {metric_name}: {metric_value*1000:.0f}ms\n"
        else:
            report += f"- {metric_name}: {metric_value:.2f}s\n"
    
    report += f"""
## Similarity Statistics
- Mean similarity: {stats['mean']:.3f}
- Median similarity: {stats['median']:.3f}
- Std deviation: {stats['std']:.3f}
- Max similarity: {stats['max']:.3f}

## Match Quality
- Excellent matches (>0.8): {stats['excellent_matches']:,}
- Good matches (0.65-0.8): {stats['good_matches']:,}
- Moderate matches (0.5-0.65): {stats['moderate_matches']:,}
- Weak matches (<0.5): {stats['weak_matches']:,}

## Entity Coverage
- Buyers with good match: {stats['buyers_with_good_match']:,} ({stats['buyers_with_good_match']/stats['n_buyers']*100:.1f}%)
- Sellers with good match: {stats['sellers_with_good_match']:,} ({stats['sellers_with_good_match']/stats['n_sellers']*100:.1f}%)
- Buyers with no match: {stats['buyers_no_match']:,} ({stats['buyers_no_match']/stats['n_buyers']*100:.1f}%)
- Sellers with no match: {stats['sellers_no_match']:,} ({stats['sellers_no_match']/stats['n_sellers']*100:.1f}%)
"""
    
    return report

# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'analyze_similarity_matrix',
    'display_similarity_statistics',
    'plot_similarity_distribution',
    'plot_top_matches_heatmap',
    'get_top_global_matches',
    'analyze_tag_importance_by_level',
    'plot_tag_importance',
    'create_network_viz',
    'export_performance_report'
]
