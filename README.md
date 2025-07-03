# tgTrax - Telegram Activity Analyzer

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Professional](https://img.shields.io/badge/code%20style-professional-black)](https://github.com/psf/black)

**tgTrax** is an enterprise-grade Python framework for comprehensive Telegram user activity analysis, specifically designed for digital forensics, behavioral analysis, and social network research. The system provides advanced temporal correlation analytics, community detection algorithms, and professional visualization capabilities.

## Architecture Overview

tgTrax implements a modular architecture comprising five core components:

- **Activity Tracking Engine**: Real-time monitoring of Telegram user online/offline status with configurable polling intervals
- **Temporal Analysis Module**: Advanced statistical correlation analysis using Spearman coefficients and Jaccard indices
- **Community Detection System**: Implementation of the Louvain method for user relationship clustering
- **Visualization Engine**: Professional Gantt chart generation for forensic timeline analysis
- **Data Persistence Layer**: High-performance SQLite backend with optimized query capabilities

## Core Features

### Data Collection
- **Real-time Status Monitoring**: Continuous tracking of user online/offline states
- **Event-driven Architecture**: Asynchronous processing using Telethon event handlers
- **Configurable Polling**: Customizable polling intervals with rate limiting
- **Robust Error Handling**: Comprehensive exception management and connection recovery

### Analytics Capabilities
- **Correlation Analysis**: Spearman rank correlation and Jaccard similarity indices
- **Community Detection**: Louvain algorithm implementation for relationship clustering
- **Temporal Pattern Recognition**: Advanced time-series analysis with configurable resampling
- **Statistical Significance Testing**: Configurable correlation thresholds with confidence intervals

### Visualization and Reporting
- **Interactive Gantt Charts**: Professional timeline visualizations using Plotly
- **Correlation Matrices**: Comprehensive user relationship mapping
- **Community Graphs**: Network topology visualization with NetworkX
- **Exportable Reports**: HTML and data export capabilities

## System Requirements

### Minimum Requirements
- **Python**: 3.9 or higher
- **Memory**: 512MB RAM minimum, 2GB recommended
- **Storage**: 100MB available disk space
- **Network**: Stable internet connection for Telegram API access

### Dependencies
- **telethon**: Telegram API client library (≥1.28.5)
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scipy**: Scientific computing and statistics
- **networkx**: Network analysis algorithms
- **python-louvain**: Community detection implementation
- **plotly**: Interactive visualization
- **rich**: Professional terminal user interface
- **python-dotenv**: Environment configuration management

## Installation and Configuration

### Standard Installation

```bash
# Clone the repository
git clone <repository-url>
cd tgTrax

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Automated Setup

```bash
# Use the provided setup script
./run.sh setup
```

### API Configuration

1. **Obtain Telegram API Credentials**:
   - Navigate to [https://my.telegram.org/apps](https://my.telegram.org/apps)
   - Create a new application
   - Record your `API_ID` and `API_HASH`

2. **Configure Environment Variables**:
   ```bash
   # The system automatically creates a template .env file
   # Edit .env with your credentials:
   TELEGRAM_API_ID=your_api_id_here
   TELEGRAM_API_HASH=your_api_hash_here
   TELEGRAM_PHONE_NUMBER=+1234567890  # Optional
   ```

3. **Application Configuration**:
   ```json
   {
     "telegram_session_name": "tgTrax_session",
     "database_path": "tgTrax.db",
     "log_level": "INFO",
     "target_users": ["username1", "username2"],
     "analysis": {
       "resample_period": "1min",
       "correlation_threshold": 0.6,
       "jaccard_threshold": 0.18
     }
   }
   ```

## Usage Guide

### Command Line Interface

#### Activity Tracking
```bash
# Start activity tracking
./run.sh start

# Start tracking in foreground (verbose mode)
./run.sh -v start

# Start tracking specific users
./run.sh start --users user1 user2 user3

# Check tracking status
./run.sh status

# Stop tracking
./run.sh stop

# View logs
./run.sh logs           # Last 20 lines
./run.sh logs -f        # Live monitoring
./run.sh logs -n 50     # Last 50 lines
```

#### Data Analysis
```bash
# Run correlation analysis
./run.sh analyze

# Run analysis with demo data
./run.sh analyze --demo

# Verbose analysis mode
./run.sh -v analyze
```

### Python API

#### Direct Module Usage
```python
from core.tracker import CorrelationTracker
from core.analysis import TemporalAnalyzer
from core.database import SQLiteDatabase

# Initialize tracking
tracker = CorrelationTracker(
    target_usernames=['user1', 'user2'],
    db_path='activity.db',
    session_name='my_session'
)

# Start tracking (asynchronous)
await tracker.start_tracking()

# Perform analysis
db = SQLiteDatabase('activity.db')
data = db.get_all_activity_for_users(['user1', 'user2'])

analyzer = TemporalAnalyzer(
    activity_df=processed_data,
    correlation_threshold=0.6
)

correlations = analyzer.get_correlation_matrix()
communities = analyzer.get_communities()
```

## Advanced Configuration

### Tracking Parameters
```json
{
  "scan_user_batch_size": 10,
  "scan_user_delay_seconds": 5,
  "USER_STATUS_POLL_INTERVAL_SECONDS": 60,
  "MINIMUM_ASSUMED_ONLINE_DURATION_SECONDS": 60
}
```

### Analysis Parameters
```json
{
  "analysis": {
    "resample_period": "1min",
    "correlation_threshold": 0.6,
    "jaccard_threshold": 0.18,
    "correlation_method": "spearman"
  }
}
```

## Output and Visualization

### Generated Files
- **tgTrax.db**: SQLite database containing all activity records
- **activity_timeline.html**: Interactive Gantt chart visualization
- **tgTrax.log**: Comprehensive application logs
- **sessions/**: Telegram session files

### Data Export Formats
- **CSV**: Raw activity data export
- **JSON**: Structured correlation results
- **HTML**: Interactive visualizations
- **SQLite**: Complete database export

## Security and Privacy

### Data Protection
- All data is stored locally in encrypted SQLite databases
- Telegram sessions use secure authentication tokens
- No sensitive data is transmitted to external services
- Comprehensive audit logging for compliance

### Rate Limiting
- Automatic API rate limit compliance
- Configurable delay parameters
- Intelligent backoff strategies
- Connection pooling optimization

## Performance Optimization

### Database Optimization
- Indexed timestamp and username columns
- Efficient UPSERT operations
- Connection pooling
- Query optimization

### Memory Management
- Streaming data processing for large datasets
- Configurable batch sizes
- Garbage collection optimization
- Memory-mapped file operations

## Troubleshooting

### Common Issues

**Authentication Errors**
- Verify API credentials in `.env` file
- Ensure phone number format is correct
- Check 2FA configuration

**Connection Issues**
- Verify internet connectivity
- Check firewall configurations
- Review proxy settings if applicable

**Performance Issues**
- Adjust polling intervals
- Optimize batch sizes
- Monitor system resources

### Logging and Diagnostics
```bash
# Enable debug logging
./run.sh -v analyze

# Check system status
./run.sh status

# Monitor logs in real-time
./run.sh logs -f
```

## Development and Contributing

### Code Standards
- **Type Hints**: Comprehensive type annotations
- **Documentation**: Detailed docstrings for all functions
- **Error Handling**: Robust exception management
- **Testing**: Unit and integration test coverage
- **Code Style**: Professional formatting and structure

### Architecture Principles
- **Modularity**: Clear separation of concerns
- **Scalability**: Designed for enterprise deployment
- **Maintainability**: Clean, documented, and tested code
- **Extensibility**: Plugin-ready architecture

## Legal and Compliance

### Terms of Use
- This software is intended for legitimate research and forensic analysis
- Users must comply with applicable laws and regulations
- Respect privacy and terms of service of monitored platforms
- Obtain appropriate permissions before monitoring activities

### Data Retention
- Implement appropriate data retention policies
- Ensure compliance with data protection regulations
- Provide mechanisms for data deletion and export
- Maintain audit logs for compliance reporting

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for complete terms and conditions.

## Support and Maintenance

### Professional Support
- Enterprise support packages available
- Custom development services
- Training and consultation
- 24/7 technical assistance options

### Community Resources
- Documentation: Comprehensive user guides and API documentation
- Issue Tracking: GitHub issues for bug reports and feature requests
- Knowledge Base: Common solutions and best practices

---

**Enterprise Deployment Ready** | **Professional Grade** | **Security Focused**
