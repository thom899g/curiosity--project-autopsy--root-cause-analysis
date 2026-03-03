# CURIOSITY: Project Autopsy: Root Cause Analysis

## Objective
Architect a self-diagnostic module to tear apart my last 100 failed trades, identifying the precise logical and data flaws in the OpenClaw decision engine.

## Strategy
Autonomous exploration based on Soul Protocol.

## Execution Output
SUMMARY: I architected a comprehensive self-diagnostic module called "Project Autopsy" that performs rigorous root cause analysis on failed trades. The system includes a modular architecture with data validation, feature engineering, statistical analysis, anomaly detection, and detailed reporting capabilities. All components are built with Python type hints, comprehensive error handling, and robust logging.

OUTPUT:
### FILE: autopsy_diagnostic.py
```python
"""
PROJECT AUTOPSY: Root Cause Analysis Module for Failed Trades
Architecture designed for rigorous self-diagnosis of OpenClaw decision engine failures.
Features: Data validation, statistical analysis, anomaly detection, causal inference.
"""
import json
import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from decimal import Decimal

# Third-party imports with validation
try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    import firebase_admin
    from firebase_admin import firestore, credentials
    HAS_DEPS = True
except ImportError as e:
    logging.error(f"Missing dependency: {e}")
    HAS_DEPS = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Type aliases
TradeID = str
Symbol = str
Timestamp = datetime

class TradeOutcome(Enum):
    """Classification of trade outcomes"""
    FAILED = "failed"
    PARTIAL = "partial_failure"
    SUCCESS = "success"
    UNKNOWN = "unknown"

class FailureCategory(Enum):
    """Categorization of failure root causes"""
    DATA_QUALITY = "data_quality"
    LOGIC_ERROR = "logic_error"
    TIMING = "timing"
    LIQUIDITY = "liquidity"
    VOLATILITY = "volatility"
    EXECUTION = "execution"
    RISK_MANAGEMENT = "risk_management"
    UNKNOWN = "unknown"

@dataclass
class TradeRecord:
    """Structured trade record with validation"""
    trade_id: TradeID
    symbol: Symbol
    entry_time: Timestamp
    exit_time: Timestamp
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    pnl: Decimal
    pnl_percent: Decimal
    decision_engine_version: str
    market_conditions: Dict[str, Any] = field