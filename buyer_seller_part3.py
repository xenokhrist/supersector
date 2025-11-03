"""
Part 3: Vectorized Similarity Calculator

CORRECT PRIORITY: T3 (40%) > T1 (30%) > T2 (20%) > Main (10%)
FOCUS: Product-level matching (T3 is most important)

NEW FEATURES:
- Track matched tags for explainability
- Support for Exhibited products column

OPTIMIZATIONS:
- Pure NumPy/SciPy operations (no Python loops)
- Sparse matrix operations for memory efficiency  
- Parallel processing for large datasets
- Batch processing with progress tracking
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# =============================================================================
# SIMILARITY CALCULATOR (VECTORIZED) WITH MATCHED TAGS TRACKING
# =============================================================================

class VectorizedSimilarityCalculator:
    """
    Ultra-fast similarity calculation using pure vectorized operations
    
    PRIORITY ORDER: T3 > T1 > T2 > Main
    - T3 (Products): 40% weight - Most specific matching
    - T1 (Categories): 30% weight - High-level alignment
    - T2 (Sub-categories): 20% weight - Medium specificity
    - Main (General): 10% weight - Background matching
    """
    
    def __init__(
        self,
        buyer_hierarchical_tags: Dict[str, List[List[str]]],
        seller_hierarchical_tags: Dict[str, List[List[str]]],
        weights: Dict[str, float] = None,
        include_exhibited: bool = False,
        seller_exhibited_tags: List[List[str]] = None
    ):
        """
        Initialize calculator with hierarchical tags
        
        Args:
            buyer_hierarchical_tags: Dict with keys 'main', 'T1', 'T2', 'T3'
            seller_hierarchical_tags: Dict with keys 'main', 'T1', 'T2', 'T3'
            weights: Custom weights (default: T3=0.4, T1=0.3, T2=0.2, main=0.1)
            include_exhibited: Whether to include exhibited products
            seller_exhibited_tags: Tags from exhibited products column
        """
        
        self.buyer_tags = buyer_hierarchical_tags
        self.seller_tags = seller_hierarchical_tags
        
        # Store original tags for matching analysis
        self.buyer_tags_original = buyer_hierarchical_tags.copy()
        self.seller_tags_original = seller_hierarchical_tags.copy()
        
        # Handle exhibited products
        self.include_exhibited = include_exhibited
        if include_exhibited and seller_exhibited_tags:
            st.info("✅ Including Exhibited Products column in matching (added to T3)")
            # Merge exhibited products into T3 (products level)
            for i, exhibited in enumerate(seller_exhibited_tags):
                if i < len(self.seller_tags['T3']):
                    # Combine T3 and exhibited products
                    combined = list(set(self.seller_tags['T3'][i] + exhibited))
                    self.seller_tags['T3'][i] = combined
        
        # Default weights prioritize product matching (T3)
        self.weights = weights or {
            'T3': 0.40,    # HIGHEST - Product level
            'T1': 0.30,    # HIGH - Category level
            'T2': 0.20,    # MEDIUM - Sub-category level  
            'main': 0.10   # LOW - General tags
        }
        
        self.n_buyers = len(buyer_hierarchical_tags['main'])
        self.n_sellers = len(seller_hierarchical_tags['main'])
        
        # Store TF-IDF vectorizers and matrices
        self.tfidf_vectorizers = {}
        self.buyer_tfidf_matrices = {}
        self.seller_tfidf_matrices = {}
        
    def _create_tfidf_matrix(
        self,
        tag_lists: List[List[str]],
        level: str
    ) -> Tuple[TfidfVectorizer, csr_matrix]:
        """
        Create TF-IDF matrix for a tag level
        
        Returns vectorizer and sparse matrix
        """
        
        # Convert tag lists to documents
        documents = [' '.join(tags) if tags else '' for tags in tag_lists]
        
        # Configure TF-IDF with level-specific parameters
        if level == 'T3':  # Product level - most specific
            max_features = 5000
            ngram_range = (1, 3)  # Include trigrams for product names
            min_df = 1
        elif level == 'T1':  # Category level
            max_features = 3000
            ngram_range = (1, 2)
            min_df = 1
        elif level == 'T2':  # Sub-category level
            max_features = 4000
            ngram_range = (1, 2)
            min_df = 1
        else:  # Main level
            max_features = 2000
            ngram_range = (1, 2)
            min_df = 1
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=0.95,
            norm='l2',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True  # Use log scaling
        )
        
        # Fit and transform
        if any(doc.strip() for doc in documents):
            tfidf_matrix = vectorizer.fit_transform(documents)
        else:
            # Empty documents - create zero matrix
            tfidf_matrix = csr_matrix((len(documents), 1))
        
        return vectorizer, tfidf_matrix
    
    def precompute_tfidf_matrices(self):
        """Precompute TF-IDF matrices for all levels"""
        
        st.info("🔄 Precomputing TF-IDF matrices for fast similarity calculation...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        levels = ['T3', 'T1', 'T2', 'main']
        
        for idx, level in enumerate(levels):
            status_text.text(f"Processing {level} level...")
            
            # Create combined vocabulary from buyers and sellers
            all_tags = self.buyer_tags[level] + self.seller_tags[level]
            all_docs = [' '.join(tags) if tags else '' for tags in all_tags]
            
            # Fit vectorizer on combined data
            vectorizer, _ = self._create_tfidf_matrix(all_tags, level)
            
            # Transform buyer and seller documents separately
            buyer_docs = [' '.join(tags) if tags else '' for tags in self.buyer_tags[level]]
            seller_docs = [' '.join(tags) if tags else '' for tags in self.seller_tags[level]]
            
            if vectorizer.vocabulary_:
                buyer_matrix = vectorizer.transform(buyer_docs)
                seller_matrix = vectorizer.transform(seller_docs)
            else:
                buyer_matrix = csr_matrix((len(buyer_docs), 1))
                seller_matrix = csr_matrix((len(seller_docs), 1))
            
            # Store
            self.tfidf_vectorizers[level] = vectorizer
            self.buyer_tfidf_matrices[level] = buyer_matrix
            self.seller_tfidf_matrices[level] = seller_matrix
            
            progress_bar.progress((idx + 1) / len(levels))
        
        progress_bar.empty()
        status_text.empty()
        
        st.success("✅ TF-IDF matrices precomputed!")
    
    def compute_similarity_matrix(self) -> np.ndarray:
        """
        Compute full similarity matrix using vectorized operations
        
        Returns: (n_buyers, n_sellers) similarity matrix
        """
        
        # Precompute TF-IDF matrices
        self.precompute_tfidf_matrices()
        
        st.info(f"🧮 Computing similarities: {self.n_buyers:,} buyers × {self.n_sellers:,} sellers = {self.n_buyers * self.n_sellers:,} combinations")
        
        # Initialize result matrix
        final_similarity = np.zeros((self.n_buyers, self.n_sellers), dtype=np.float32)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Compute cosine similarity for each level
        levels = ['T3', 'T1', 'T2', 'main']
        
        for idx, level in enumerate(levels):
            status_text.text(f"Computing {level} similarities (weight: {self.weights[level]:.0%})...")
            
            buyer_matrix = self.buyer_tfidf_matrices[level]
            seller_matrix = self.seller_tfidf_matrices[level]
            
            # Compute cosine similarity (vectorized)
            if buyer_matrix.shape[1] > 0 and seller_matrix.shape[1] > 0:
                level_similarity = cosine_similarity(buyer_matrix, seller_matrix).astype(np.float32)
            else:
                level_similarity = np.zeros((self.n_buyers, self.n_sellers), dtype=np.float32)
            
            # Add weighted contribution
            final_similarity += self.weights[level] * level_similarity
            
            progress_bar.progress((idx + 1) / len(levels))
            
            # Memory cleanup
            del level_similarity
        
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Similarity matrix computed! Shape: {final_similarity.shape}")
        
        return final_similarity
    
    def compute_similarity_matrix_parallel(self, n_workers: int = 4) -> np.ndarray:
        """
        Compute similarity matrix using parallel processing
        
        Best for very large datasets (>5M combinations)
        """
        
        # Precompute TF-IDF matrices
        self.precompute_tfidf_matrices()
        
        st.info(f"🚀 Computing similarities in parallel ({n_workers} workers)...")
        
        # For parallel processing, we'll compute each level's similarity in parallel
        final_similarity = np.zeros((self.n_buyers, self.n_sellers), dtype=np.float32)
        
        def compute_level_similarity(level: str) -> np.ndarray:
            """Compute similarity for one level"""
            buyer_matrix = self.buyer_tfidf_matrices[level]
            seller_matrix = self.seller_tfidf_matrices[level]
            
            if buyer_matrix.shape[1] > 0 and seller_matrix.shape[1] > 0:
                return cosine_similarity(buyer_matrix, seller_matrix).astype(np.float32)
            else:
                return np.zeros((self.n_buyers, self.n_sellers), dtype=np.float32)
        
        levels = ['T3', 'T1', 'T2', 'main']
        
        # Use ThreadPoolExecutor for I/O-bound operations
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(compute_level_similarity, level): level for level in levels}
            
            progress_bar = st.progress(0)
            completed = 0
            
            for future in futures:
                level = futures[future]
                level_similarity = future.result()
                
                # Add weighted contribution
                final_similarity += self.weights[level] * level_similarity
                
                completed += 1
                progress_bar.progress(completed / len(levels))
                
                # Memory cleanup
                del level_similarity
            
            progress_bar.empty()
        
        st.success("✅ Parallel similarity computation complete!")
        
        return final_similarity
    
    def get_matched_tags(self, buyer_idx: int, seller_idx: int) -> Dict[str, List[str]]:
        """
        Get matched tags between a buyer and seller
        
        Returns dict with matched tags per level
        """
        
        matched = {
            'main': [],
            'T1': [],
            'T2': [],
            'T3': [],
            'all': []
        }
        
        for level in ['main', 'T1', 'T2', 'T3']:
            if buyer_idx < len(self.buyer_tags_original[level]) and seller_idx < len(self.seller_tags_original[level]):
                buyer_tags_set = set(self.buyer_tags_original[level][buyer_idx])
                seller_tags_set = set(self.seller_tags_original[level][seller_idx])
                
                # Find intersection
                common = buyer_tags_set & seller_tags_set
                matched[level] = sorted(list(common))
                matched['all'].extend(common)
        
        # Remove duplicates from 'all'
        matched['all'] = sorted(list(set(matched['all'])))
        
        return matched

# =============================================================================
# BATCH SIMILARITY COMPUTATION (FOR MEMORY EFFICIENCY)
# =============================================================================

def compute_similarity_batched(
    calculator: VectorizedSimilarityCalculator,
    batch_size: int = 1000
) -> np.ndarray:
    """
    Compute similarity in batches to reduce memory usage
    
    Useful for very large datasets that don't fit in memory
    """
    
    n_buyers = calculator.n_buyers
    n_sellers = calculator.n_sellers
    
    # Precompute TF-IDF matrices
    calculator.precompute_tfidf_matrices()
    
    st.info(f"🔄 Computing similarities in batches (batch size: {batch_size})...")
    
    # Initialize result matrix
    similarity_matrix = np.zeros((n_buyers, n_sellers), dtype=np.float32)
    
    # Calculate number of batches
    n_buyer_batches = (n_buyers + batch_size - 1) // batch_size
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for batch_idx in range(n_buyer_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_buyers)
        
        status_text.text(f"Processing buyers {start_idx+1}-{end_idx} of {n_buyers}...")
        
        # Compute similarity for this batch of buyers
        batch_similarity = np.zeros((end_idx - start_idx, n_sellers), dtype=np.float32)
        
        for level in ['T3', 'T1', 'T2', 'main']:
            buyer_batch = calculator.buyer_tfidf_matrices[level][start_idx:end_idx]
            seller_matrix = calculator.seller_tfidf_matrices[level]
            
            if buyer_batch.shape[1] > 0 and seller_matrix.shape[1] > 0:
                level_similarity = cosine_similarity(buyer_batch, seller_matrix).astype(np.float32)
                batch_similarity += calculator.weights[level] * level_similarity
                del level_similarity
        
        # Store results
        similarity_matrix[start_idx:end_idx] = batch_similarity
        
        progress_bar.progress((batch_idx + 1) / n_buyer_batches)
    
    progress_bar.empty()
    status_text.empty()
    
    st.success("✅ Batched similarity computation complete!")
    
    return similarity_matrix

# =============================================================================
# RECOMMENDATION GENERATION WITH MATCHED TAGS
# =============================================================================

def get_top_k_recommendations(
    similarity_matrix: np.ndarray,
    entity_idx: int,
    target_df: pd.DataFrame,
    is_buyer_to_seller: bool,
    calculator: VectorizedSimilarityCalculator,
    top_k: int = 5
) -> pd.DataFrame:
    """
    Get top-K recommendations for a specific entity WITH MATCHED TAGS
    
    Args:
        similarity_matrix: Full similarity matrix
        entity_idx: Index of source entity
        target_df: DataFrame of target entities
        is_buyer_to_seller: True if buyer→seller, False if seller→buyer
        calculator: Calculator instance for matched tags
        top_k: Number of recommendations
    
    Returns:
        DataFrame with top-K recommendations, scores, and matched tags
    """
    
    # Get similarity scores
    if is_buyer_to_seller:
        if entity_idx >= similarity_matrix.shape[0]:
            return pd.DataFrame()
        scores = similarity_matrix[entity_idx, :]
        source_idx = entity_idx
    else:
        if entity_idx >= similarity_matrix.shape[1]:
            return pd.DataFrame()
        scores = similarity_matrix[:, entity_idx]
        source_idx = entity_idx
    
    # Get top-K indices
    actual_k = min(top_k, len(scores))
    
    if len(scores) > 100:
        # Use argpartition for large arrays (faster)
        top_indices = np.argpartition(-scores, actual_k-1)[:actual_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
    else:
        top_indices = np.argsort(-scores)[:actual_k]
    
    # Create results DataFrame
    results = target_df.iloc[top_indices].copy()
    results['similarity_score'] = scores[top_indices]
    results['rank'] = range(1, len(results) + 1)
    
    # Add matched tags
    matched_tags_list = []
    for target_idx in top_indices:
        if is_buyer_to_seller:
            matched = calculator.get_matched_tags(source_idx, target_idx)
        else:
            matched = calculator.get_matched_tags(target_idx, source_idx)
        
        # Format matched tags
        if matched['all']:
            matched_tags_str = ', '.join(matched['all'][:10])  # Show top 10
            if len(matched['all']) > 10:
                matched_tags_str += f" (+{len(matched['all'])-10} more)"
        else:
            matched_tags_str = "(no exact matches)"
        
        matched_tags_list.append(matched_tags_str)
    
    results['Matching_Tags'] = matched_tags_list
    
    # Move important columns to front
    cols = ['rank', 'similarity_score', 'Matching_Tags'] + [col for col in results.columns if col not in ['rank', 'similarity_score', 'Matching_Tags']]
    results = results[cols]
    
    return results.reset_index(drop=True)

# =============================================================================
# BULK EXPORT: ALL MATCHES ABOVE THRESHOLD WITH MATCHED TAGS
# =============================================================================

def export_all_matches_above_threshold(
    similarity_matrix: np.ndarray,
    buyers_df: pd.DataFrame,
    sellers_df: pd.DataFrame,
    buyer_name_col: str,
    seller_name_col: str,
    calculator: VectorizedSimilarityCalculator,
    threshold: float = 0.65,
    batch_size: int = 10000
) -> pd.DataFrame:
    """
    Export all buyer-seller matches above similarity threshold WITH MATCHED TAGS
    
    Returns DataFrame with columns:
    - Rank
    - Buyer
    - Seller
    - Score
    - Matching_Tags
    - Stand_no
    - (other seller columns)
    
    OPTIMIZED: Processes in batches to handle large result sets
    """
    
    st.info(f"🔍 Finding all matches with similarity > {threshold:.2f}...")
    
    # Find all matches above threshold using NumPy (vectorized)
    buyer_indices, seller_indices = np.where(similarity_matrix > threshold)
    scores = similarity_matrix[buyer_indices, seller_indices]
    
    n_matches = len(buyer_indices)
    
    if n_matches == 0:
        st.warning(f"⚠️ No matches found above threshold {threshold:.2f}")
        return pd.DataFrame()
    
    st.info(f"📊 Found {n_matches:,} matches above threshold {threshold:.2f}")
    st.info(f"🏷️ Computing matched tags for all matches...")
    
    # Process in batches for memory efficiency
    n_batches = (n_matches + batch_size - 1) // batch_size
    
    all_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_matches)
        
        status_text.text(f"Processing matches {start_idx+1:,}-{end_idx:,} of {n_matches:,}...")
        
        batch_buyers = buyer_indices[start_idx:end_idx]
        batch_sellers = seller_indices[start_idx:end_idx]
        batch_scores = scores[start_idx:end_idx]
        
        # Compute matched tags for this batch
        batch_matched_tags = []
        for b_idx, s_idx in zip(batch_buyers, batch_sellers):
            matched = calculator.get_matched_tags(b_idx, s_idx)
            if matched['all']:
                matched_str = ', '.join(matched['all'][:15])  # Show top 15
                if len(matched['all']) > 15:
                    matched_str += f" (+{len(matched['all'])-15} more)"
            else:
                matched_str = "(no exact matches)"
            batch_matched_tags.append(matched_str)
        
        # Create batch DataFrame
        batch_results = pd.DataFrame({
            'Buyer': buyers_df.iloc[batch_buyers][buyer_name_col].values,
            'Seller': sellers_df.iloc[batch_sellers][seller_name_col].values,
            'Score': batch_scores,
            'Matching_Tags': batch_matched_tags
        })
        
        # Add Stand_no if available
        if 'Stand_no' in sellers_df.columns:
            batch_results['Stand_no'] = sellers_df.iloc[batch_sellers]['Stand_no'].values
        
        all_results.append(batch_results)
        
        progress_bar.progress((batch_idx + 1) / n_batches)
    
    progress_bar.empty()
    status_text.empty()
    
    # Combine all batches
    final_results = pd.concat(all_results, ignore_index=True)
    
    # Sort by score descending
    final_results = final_results.sort_values('Score', ascending=False).reset_index(drop=True)
    
    # Add rank
    final_results.insert(0, 'Rank', range(1, len(final_results) + 1))
    
    st.success(f"✅ Exported {len(final_results):,} matches with matched tags!")
    
    return final_results

# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'VectorizedSimilarityCalculator',
    'compute_similarity_batched',
    'get_top_k_recommendations',
    'export_all_matches_above_threshold'
]
