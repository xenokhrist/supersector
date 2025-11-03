"""
OPTIMIZED Buyer-Seller Recommender System
Part 1: Core Utilities, Configuration, and Security

PERFORMANCE TARGET: 5-10 minutes for 10k buyers × 1.5k sellers
FOCUS: Product matching (T3 priority > T1 > T2 > Main)
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import warnings
from typing import List, Set, Optional, Tuple, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, vstack
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
import time
from datetime import datetime

warnings.filterwarnings('ignore')

# Optional imports
try:
    from pyvis.network import Network
    import networkx as nx
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Centralized configuration for the recommender system"""
    
    # File constraints
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_ROWS = 50000
    ALLOWED_EXTENSIONS = {'.xlsx', '.xls'}
    
    # Performance settings
    MAX_WORKERS = min(12, multiprocessing.cpu_count())  # Use most cores
    BATCH_SIZE = 500  # For batch processing
    CACHE_SIZE = 50000  # Increased cache for 6k vocabulary
    
    # Algorithm settings - CORRECTED PRIORITIES
    DEFAULT_WEIGHTS = {
        'T3': 0.40,   # HIGHEST - Product level (most specific)
        'T1': 0.30,   # HIGH - Category level
        'T2': 0.20,   # MEDIUM - Sub-category level
        'main': 0.10  # LOW - General tags
    }
    
    # Export settings
    EXPORT_THRESHOLD = 0.65
    EXPORT_BATCH_SIZE = 10000  # Write in batches for large exports
    
    # TF-IDF settings
    TFIDF_MAX_FEATURES = 10000  # Large vocabulary support
    TFIDF_MIN_DF = 1
    TFIDF_MAX_DF = 0.95
    TFIDF_NGRAM_RANGE = (1, 2)
    
    # UI settings
    HEATMAP_MAX_DISPLAY = 50
    NETWORK_MAX_NODES = 150
    NETWORK_MAX_EDGES = 300
    
    # Security patterns
    DANGEROUS_PATTERNS = [
        r'<script', r'javascript:', r'vbscript:', r'onload=', r'onerror=',
        r'<iframe', r'<object', r'<embed', r'eval\(', r'exec\('
    ]

# =============================================================================
# SECURITY & VALIDATION
# =============================================================================

def validate_file_security(file_obj) -> Tuple[bool, str]:
    """Validate uploaded file for security"""
    if file_obj is None:
        return False, "No file provided"
    
    if hasattr(file_obj, 'size') and file_obj.size > Config.MAX_FILE_SIZE:
        return False, f"File too large. Max: {Config.MAX_FILE_SIZE // (1024*1024)}MB"
    
    if not any(file_obj.name.lower().endswith(ext) for ext in Config.ALLOWED_EXTENSIONS):
        return False, f"Invalid file type. Allowed: {Config.ALLOWED_EXTENSIONS}"
    
    return True, "Valid"

def sanitize_text(text: Any) -> str:
    """Sanitize text input for security"""
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text).strip()
    
    # Remove dangerous patterns
    for pattern in Config.DANGEROUS_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            text = re.sub(pattern, '[FILTERED]', text, flags=re.IGNORECASE)
    
    # Keep only printable characters
    text = ''.join(char for char in text if char.isprintable() or char.isspace())
    
    return text

def validate_dataframe_size(df: pd.DataFrame, name: str) -> bool:
    """Validate DataFrame size constraints"""
    if len(df) > Config.MAX_ROWS:
        st.error(f"❌ {name} too large. Max rows: {Config.MAX_ROWS:,}")
        return False
    if len(df) == 0:
        st.error(f"❌ {name} is empty")
        return False
    return True

def check_columns_exist(df: pd.DataFrame, required_cols: List[str], df_name: str) -> Tuple[bool, List[str]]:
    """Check if required columns exist in DataFrame"""
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ {df_name} missing columns: {missing_cols}")
        st.info(f"Available columns: {list(df.columns)}")
        return False, missing_cols
    
    return True, []

# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

class PerformanceMonitor:
    """Track performance metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        
    def start(self, name: str = "total"):
        """Start timing"""
        self.start_time = time.time()
        self.metrics[name] = {'start': self.start_time}
        
    def end(self, name: str = "total") -> float:
        """End timing and return duration"""
        if name not in self.metrics:
            return 0.0
        
        duration = time.time() - self.metrics[name]['start']
        self.metrics[name]['duration'] = duration
        return duration
    
    def get_summary(self) -> Dict[str, float]:
        """Get performance summary"""
        return {k: v.get('duration', 0.0) for k, v in self.metrics.items()}
    
    def display_metrics(self):
        """Display metrics in Streamlit sidebar"""
        st.sidebar.markdown("---")
        st.sidebar.header("⚡ Performance Metrics")
        
        for name, data in self.metrics.items():
            if 'duration' in data:
                duration = data['duration']
                if duration < 1:
                    st.sidebar.metric(name.title(), f"{duration*1000:.0f}ms")
                else:
                    st.sidebar.metric(name.title(), f"{duration:.2f}s")

# =============================================================================
# DATA LOADING
# =============================================================================

def load_excel_optimized(file_obj, name: str) -> Optional[pd.DataFrame]:
    """Load Excel file with optimization"""
    try:
        # Security check
        is_valid, msg = validate_file_security(file_obj)
        if not is_valid:
            st.error(f"❌ {name} file error: {msg}")
            return None
        
        # Load with optimization
        df = pd.read_excel(
            file_obj,
            engine='openpyxl',
            dtype=str  # Load as string to avoid type issues
        )
        
        # Validate size
        if not validate_dataframe_size(df, name):
            return None
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Sanitize text columns
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(sanitize_text)
        
        st.success(f"✅ Loaded {len(df):,} {name.lower()} records")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Failed to load {name}: {str(e)}")
        return None

# =============================================================================
# SAMPLE DATA GENERATOR
# =============================================================================

def create_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create sample data for testing"""
    
    buyers = pd.DataFrame({
        'BuyerID': ['B1', 'B2', 'B3', 'B4', 'B5'],
        'buyer_company': [
            'Global Food Corp',
            'TechElectronics Ltd',
            'Fashion Forward LLC',
            'AutoParts International',
            'Green Energy Solutions'
        ],
        'buyer_tag': [
            'food, beverage, packaged goods',
            'electronics, consumer tech, gadgets',
            'apparel, fashion, accessories',
            'automotive, parts, components',
            'renewable energy, solar, wind'
        ],
        'buyer_T1': [
            'food processing, beverage manufacturing',
            'electronics manufacturing, assembly',
            'textile, garment manufacturing',
            'automotive manufacturing, assembly',
            'energy systems, power generation'
        ],
        'buyer_T2': [
            'organic food, natural ingredients, packaging',
            'consumer electronics, mobile devices, computers',
            'fashion design, retail, distribution',
            'engine parts, brake systems, electrical',
            'solar panels, wind turbines, batteries'
        ],
        'buyer_T3': [
            'organic flour, natural sweeteners, eco packaging',
            'smartphones, laptops, tablets, smart watches',
            'dresses, shirts, pants, shoes, bags',
            'brake pads, spark plugs, alternators, batteries',
            'solar panels 300W, wind turbines 5kW, lithium batteries'
        ]
    })
    
    sellers = pd.DataFrame({
        'SellerID': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'],
        'Company name': [
            'Organic Ingredients Inc',
            'Advanced Electronics Co',
            'Textile Innovations Ltd',
            'Precision Auto Parts',
            'Solar Tech Manufacturing',
            'Quality Food Packaging'
        ],
        'Stand_no': ['A101', 'B205', 'C330', 'D112', 'E445', 'A105'],
        'tag': [
            'organic, food ingredients, natural',
            'electronics, components, semiconductors',
            'textiles, fabrics, sustainable',
            'automotive, precision parts, quality',
            'solar, renewable, clean energy',
            'packaging, food-grade, eco-friendly'
        ],
        'T1': [
            'food ingredients, organic certification',
            'electronic components, circuit boards',
            'textile manufacturing, fabric treatment',
            'automotive components, precision machining',
            'solar systems, renewable energy',
            'food packaging, materials'
        ],
        'T2': [
            'organic flour, natural preservatives, sweeteners',
            'microchips, sensors, displays, batteries',
            'cotton fabric, synthetic materials, dyes',
            'brake components, engine parts, electrical',
            'photovoltaic panels, inverters, mounting',
            'biodegradable packaging, food containers'
        ],
        'T3': [
            'organic wheat flour, stevia, monk fruit extract',
            'ARM processors, OLED displays, Li-ion batteries',
            'organic cotton, recycled polyester, natural dyes',
            'ceramic brake pads, iridium spark plugs, AGM batteries',
            'monocrystalline solar panels 300W, micro inverters',
            'compostable food containers, recycled cardboard boxes'
        ]
    })
    
    return buyers, sellers

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_get_column(df: pd.DataFrame, col: str, default: Any = "") -> pd.Series:
    """Safely get column from DataFrame"""
    if col in df.columns:
        return df[col].fillna(default)
    return pd.Series([default] * len(df), index=df.index)

def format_number(num: int) -> str:
    """Format large numbers with commas"""
    return f"{num:,}"

def estimate_processing_time(n_buyers: int, n_sellers: int) -> str:
    """Estimate processing time based on dataset size"""
    combinations = n_buyers * n_sellers
    
    if combinations < 1_000_000:
        return "< 2 minutes"
    elif combinations < 5_000_000:
        return "2-5 minutes"
    elif combinations < 10_000_000:
        return "5-8 minutes"
    else:
        return "8-15 minutes"

def get_memory_usage(df: pd.DataFrame) -> str:
    """Get DataFrame memory usage"""
    mem_bytes = df.memory_usage(deep=True).sum()
    
    if mem_bytes < 1024:
        return f"{mem_bytes} B"
    elif mem_bytes < 1024**2:
        return f"{mem_bytes/1024:.1f} KB"
    elif mem_bytes < 1024**3:
        return f"{mem_bytes/(1024**2):.1f} MB"
    else:
        return f"{mem_bytes/(1024**3):.1f} GB"

# =============================================================================
# EXPORT UTILITIES
# =============================================================================

def create_export_filename(prefix: str, extension: str = "csv") -> str:
    """Create timestamped export filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"

def prepare_export_buffer(df: pd.DataFrame, format: str = "csv") -> io.BytesIO:
    """Prepare export buffer for download"""
    buffer = io.BytesIO()
    
    if format == "csv":
        df.to_csv(buffer, index=False, encoding='utf-8-sig')
    elif format == "excel":
        df.to_excel(buffer, index=False, engine='openpyxl')
    
    buffer.seek(0)
    return buffer

# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'perf_monitor' not in st.session_state:
        st.session_state.perf_monitor = PerformanceMonitor()
    
    if 'similarity_matrix' not in st.session_state:
        st.session_state.similarity_matrix = None
    
    if 'processed_buyers' not in st.session_state:
        st.session_state.processed_buyers = None
    
    if 'processed_sellers' not in st.session_state:
        st.session_state.processed_sellers = None
    
    if 'all_matches_cache' not in st.session_state:
        st.session_state.all_matches_cache = None

# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'Config',
    'PerformanceMonitor',
    'validate_file_security',
    'sanitize_text',
    'validate_dataframe_size',
    'check_columns_exist',
    'load_excel_optimized',
    'create_sample_data',
    'safe_get_column',
    'format_number',
    'estimate_processing_time',
    'get_memory_usage',
    'create_export_filename',
    'prepare_export_buffer',
    'initialize_session_state',
    'PYVIS_AVAILABLE'
]
