"""
Pydantic models for tgTrax application.

This module provides data validation and serialization models using Pydantic
to ensure type safety and data integrity throughout the application.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator, ConfigDict


class TelegramConfig(BaseModel):
    """Configuration model for Telegram API settings."""
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    api_id: int = Field(..., gt=0, description="Telegram API ID")
    api_hash: str = Field(..., min_length=32, max_length=32, description="Telegram API Hash")
    phone_number: Optional[str] = Field(None, description="Phone number for authentication")
    session_name: str = Field(..., min_length=1, description="Session file name")
    
    @validator('phone_number')
    def validate_phone_number(cls, v):
        if v and not v.startswith('+'):
            raise ValueError('Phone number must start with +')
        return v


class DatabaseConfig(BaseModel):
    """Configuration model for database settings."""
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    db_path: str = Field(..., min_length=1, description="Database file path")
    timeout_seconds: float = Field(15.0, gt=0, le=3600, description="Connection timeout in seconds")
    
    
class TrackingConfig(BaseModel):
    """Configuration model for user tracking settings."""
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    target_usernames: List[str] = Field(..., min_items=1, description="List of usernames to track")
    poll_interval_seconds: int = Field(60, ge=10, le=3600, description="Polling interval in seconds")
    min_online_duration_seconds: int = Field(60, ge=0, le=7200, description="Minimum assumed online duration")
    max_users_before_delay: int = Field(10, ge=1, le=1000, description="Max users before applying delay")
    inter_user_delay_seconds: float = Field(0.5, ge=0, le=60, description="Delay between user polls")
    
    @validator('target_usernames')
    def validate_usernames(cls, v):
        # Strip @ symbols and validate usernames
        cleaned = []
        for username in v:
            clean_username = username.strip().lstrip('@')
            if not clean_username:
                raise ValueError('Username cannot be empty after cleaning')
            if len(clean_username) < 5 or len(clean_username) > 32:
                raise ValueError(f'Username {clean_username} must be 5-32 characters')
            cleaned.append(clean_username)
        return cleaned


class ActivityRecord(BaseModel):
    """Model for user activity records."""
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    username: str = Field(..., min_length=1, max_length=32, description="Username")
    timestamp: int = Field(..., description="Unix timestamp in milliseconds")
    online: bool = Field(..., description="Online status")
    source: str = Field("unknown", description="Source of the activity record")
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        if v < 0:
            raise ValueError('Timestamp cannot be negative')
        # Check if timestamp is reasonable (not too far in past/future)
        now_ms = int(datetime.now().timestamp() * 1000)
        one_year_ms = 365 * 24 * 60 * 60 * 1000
        if v < (now_ms - one_year_ms) or v > (now_ms + one_year_ms):
            raise ValueError('Timestamp is outside reasonable range')
        return v


class UserStatus(BaseModel):
    """Model for user status information."""
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    user_id: int = Field(..., gt=0, description="Telegram user ID")
    username: str = Field(..., min_length=1, max_length=32, description="Username")
    is_online: bool = Field(..., description="Current online status")
    last_seen: Optional[datetime] = Field(None, description="Last seen timestamp")
    status_type: str = Field("unknown", description="Status type from Telegram")


class CorrelationResult(BaseModel):
    """Model for correlation analysis results."""
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    user1: str = Field(..., min_length=1, description="First user")
    user2: str = Field(..., min_length=1, description="Second user")
    correlation_value: float = Field(..., ge=-1.0, le=1.0, description="Correlation coefficient")
    method: str = Field("spearman", description="Correlation method used")
    significance_threshold: float = Field(0.6, ge=0.0, le=1.0, description="Significance threshold")
    is_significant: bool = Field(..., description="Whether correlation is significant")


class AnalysisConfig(BaseModel):
    """Configuration model for analysis settings."""
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    resample_period: str = Field("1min", description="Resampling period for analysis")
    correlation_threshold: float = Field(0.6, ge=0.0, le=1.0, description="Correlation threshold")
    jaccard_threshold: float = Field(0.18, ge=0.0, le=1.0, description="Jaccard index threshold")
    correlation_method: str = Field("spearman", description="Correlation method")
    
    @validator('resample_period')
    def validate_resample_period(cls, v):
        # Basic validation for pandas resample periods
        import re
        if not re.match(r'^\d+[SMHD]?$|^\d+(min|mins)$', v):
            raise ValueError(f'Invalid resample period: {v}')
        return v


class ErrorInfo(BaseModel):
    """Model for structured error information."""
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    error_type: str = Field(..., min_length=1, description="Error type/category")
    message: str = Field(..., min_length=1, description="Error message")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When error occurred")
    severity: str = Field("error", regex="^(debug|info|warning|error|critical)$", description="Error severity")


class TrackerState(BaseModel):
    """Model for tracker state information."""
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    is_running: bool = Field(False, description="Whether tracker is running")
    connected_users: List[str] = Field(default_factory=list, description="Currently connected users")
    last_poll_time: Optional[datetime] = Field(None, description="Last polling time")
    total_activity_records: int = Field(0, ge=0, description="Total activity records stored")
    errors_count: int = Field(0, ge=0, description="Number of errors encountered")
    uptime_seconds: float = Field(0.0, ge=0.0, description="Uptime in seconds")