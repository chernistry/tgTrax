# tgTrax - Telegram Activity Analyzer

<p align="center">
  <img src="assets/logo.png" alt="tgTrax Logo" width="300"/>
</p>

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**tgTrax** is a Python-based toolkit for tracking and analyzing Telegram user activity, designed for digital forensics and behavioral analysis.

## Key Features

- **Activity Tracking:** Monitor online/offline status of Telegram users with configurable polling.
- **Temporal Analysis:** Calculate correlations (Spearman, Jaccard) and build user connection graphs.
- **Community Detection:** Identify user groups using Louvain method.
- **Gantt Charts:** Visualize activity timelines for forensic analysis.
- **SQLite Backend:** Store and query activity data efficiently.

## Quick Start

1. **Installation:**
   ```bash
   git clone <repository-url>
   cd tgTrax
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configuration:**
   - Set `TELEGRAM_API_ID` and `TELEGRAM_API_HASH` in `.env`.
   - Configure target users and thresholds in `tgTrax_config.json`.

3. **Usage:**
   - Start tracking: `./run.sh start`
   - Analyze data: `./run.sh analyze`

## Screenshots

![User Community Detection](assets/screenshot1.png)
*User communities based on activity correlation.*

![Activity Timeline](assets/screenshot2.png)
*Gantt chart of user online sessions.*

## License

MIT License. See [LICENSE](LICENSE) for details.
