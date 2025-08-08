# ==== TEMPORAL ANALYSIS MODULE ==== #
# Description: This module provides the TemporalAnalyzer class for analyzing user activity patterns,
#              calculating correlations, detecting communities, and generating activity summaries.
#              It also includes a helper function to create Gantt charts for visualizing activity.


import datetime # Added for type hinting and usage in get_activity_intervals
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union, Optional # For precise type hinting

import networkx as nx
import numpy as np
import pandas as pd
# from scipy.stats import spearmanr  # Not used directly; pandas handles spearman

import community.community_louvain as community_louvain # Explicitly for clarity
from utils import tui # For console output


# --- Constants ---
# (No module-level constants defined after removals)


# --- TemporalAnalyzer Class ---
class TemporalAnalyzer:
    """
    Analyzes temporal patterns in user activity data, focusing on correlations
    and community structures.
    """

    def __init__(
        self,
        activity_df: pd.DataFrame,
        resample_period: str = "1min",
        correlation_threshold: float = 0.6,
        jaccard_threshold: float = 0.18,
        debug_users: Optional[List[str]] = None,
    ):
        """
        Initializes the TemporalAnalyzer with activity data.

        Args:
            activity_df: DataFrame with a DateTimeIndex, columns for each user,
                and boolean/int (0/1) values for online status.
            resample_period: Pandas resampling period string (e.g., '1T' for
                1 minute, '5T' for 5 minutes).
            correlation_threshold: Default threshold for correlation significance.
            jaccard_threshold: Default threshold for Jaccard significance.
            debug_users: Optional usernames to emit compact debug samples for.
        """
        self.default_correlation_threshold: float = correlation_threshold
        self.default_jaccard_threshold: float = jaccard_threshold  # New attribute
        self.df_resampled: pd.DataFrame
        self.correlation_matrix: pd.DataFrame
        self.jaccard_matrix: pd.DataFrame
        self.user_list: List[str]
        self.resample_offset: pd.DateOffset | pd.Timedelta | None = None
        self.resample_seconds: float | None = None
        self.debug_users: List[str] = debug_users or []

        if activity_df.empty:
            tui.tui_print_warning(
                "TemporalAnalyzer initialized with an empty DataFrame."
            )
            self.df_resampled = pd.DataFrame()
            self.correlation_matrix = pd.DataFrame()
            self.jaccard_matrix = pd.DataFrame()  # Initialize empty
            self.user_list = []
            return

        self.user_list = activity_df.columns.tolist()
        activity_df_numeric: pd.DataFrame = activity_df.astype(float)

        # Resample: Take the max to see if user was online at all during the period.
        # fillna(0) assumes offline if no data point in interval.
        self.df_resampled = (
            activity_df_numeric.resample(resample_period).max().fillna(0)
        )
        tui.tui_print_info(
            f"Activity data resampled to {resample_period}. "
            f"Shape: {self.df_resampled.shape}"
        )

        # Optional compact debug output for selected users
        if self.debug_users:
            for dbg_user in self.debug_users:
                if dbg_user in self.df_resampled.columns:
                    series_dbg = self.df_resampled[dbg_user]
                    tui.tui_print_debug(
                        f"Resampled sample for {dbg_user} (head5):\n{series_dbg.head(5).to_string()}"
                    )
                    tui.tui_print_debug(
                        f"Resampled sample for {dbg_user} (tail5):\n{series_dbg.tail(5).to_string()}"
                    )

        # Cache frequency info once
        self._infer_resample_offset_and_seconds()

        self.correlation_matrix = self._calculate_correlations()
        self.jaccard_matrix = self._calculate_jaccard_indices()

    def _infer_resample_offset_and_seconds(self) -> None:
        """
        Infers and caches the resample offset and seconds for downstream use.
        """
        if self.df_resampled.empty:
            self.resample_offset = None
            self.resample_seconds = None
            return

        freq_offset: pd.DateOffset | None = self.df_resampled.index.freq
        if freq_offset is None:
            inferred = pd.infer_freq(self.df_resampled.index)
            if inferred:
                try:
                    freq_offset = pd.tseries.frequencies.to_offset(inferred)
                except Exception:
                    freq_offset = None

        if freq_offset is None:
            # Fallback 1 minute step if not inferable
            freq_offset = pd.Timedelta(minutes=1)
            tui.tui_print_warning(
                "Could not infer resample frequency, defaulting to 1 minute."
            )

        self.resample_offset = freq_offset
        # Convert to seconds regardless of offset type
        if hasattr(freq_offset, "nanos"):
            self.resample_seconds = float(getattr(freq_offset, "nanos")) / 1e9
        else:
            self.resample_seconds = pd.to_timedelta(freq_offset).total_seconds()


    def _calculate_correlations(self, method: str = "spearman") -> pd.DataFrame:
        """
        Calculates the correlation matrix between users\' online statuses.

        Args:
            method: The correlation method to use (e.g., \'spearman\', \'pearson\').
                    Spearman is generally good for non-linear relationships and
                    ordinal data.

        Returns:
            A pandas DataFrame representing the correlation matrix. Returns an
            empty DataFrame if data is insufficient.
        """
        if self.df_resampled.empty or len(self.df_resampled.columns) < 2:
            tui.tui_print_warning(
                "Not enough data or users to calculate correlations."
            )
            return pd.DataFrame()

        corr_matrix: pd.DataFrame = self.df_resampled.corr(method=method)
        tui.tui_print_info(f"Correlation matrix calculated using {method} method.")
        # tui.tui_print_debug(f"Correlation Matrix:\\n{corr_matrix.to_string()}") # Verbose
        return corr_matrix


    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Retrieves the calculated correlation matrix.

        Returns:
            A pandas DataFrame with user-to-user correlations.
        """
        return self.correlation_matrix


    def get_significant_pairs(
        self, threshold: float | None = None
    ) -> List[Tuple[Tuple[str, str], float]]:
        """
        Identifies pairs of users with correlation above a given threshold.

        Args:
            threshold: The correlation coefficient threshold. If None, the
                       instance\'s default_correlation_threshold is used.
                       The absolute value of the correlation is compared.

        Returns:
            A list of tuples, where each tuple contains:
            ((user1, user2), correlation_value).
            The list is sorted by the absolute correlation value in descending order.
            Returns an empty list if the correlation matrix is empty.
        """
        current_threshold: float = (
            threshold
            if threshold is not None
            else self.default_correlation_threshold
        )

        if self.correlation_matrix.empty:
            return []

        significant_pairs: List[Tuple[Tuple[str, str], float]] = []
        columns: List[str] = self.correlation_matrix.columns.tolist()

        # Iterate over the upper triangle of the matrix
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                user1: str = columns[i]
                user2: str = columns[j]
                correlation_value: float = self.correlation_matrix.iloc[i, j]

                if abs(correlation_value) >= current_threshold:
                    significant_pairs.append(((user1, user2), correlation_value))

        significant_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        tui.tui_print_info(
            f"Found {len(significant_pairs)} significant pairs "
            f"with threshold >= {abs(current_threshold):.2f}"
        )
        return significant_pairs


    def build_correlation_graph(
        self, threshold: float | None = None
    ) -> nx.Graph:
        """
        Creates a NetworkX graph based on significant user correlations.

        Nodes represent users, and edges represent significant correlations,
        weighted by the correlation coefficient. Community detection (Louvain)
        is applied if edges exist.

        Args:
            threshold: The correlation threshold for including an edge. If None,
                       the instance\'s default_correlation_threshold is used.

        Returns:
            A NetworkX graph. Nodes will have a \'community\' attribute.
        """
        current_threshold: float = (
            threshold
            if threshold is not None
            else self.default_correlation_threshold
        )
        
        graph = nx.Graph()
        if self.df_resampled.empty or not self.user_list:
            tui.tui_print_warning(
                "Cannot build graph: No resampled data or user list."
            )
            return graph

        for user in self.user_list:
            graph.add_node(user)

        significant_pairs = self.get_significant_pairs(current_threshold)

        for pair_info in significant_pairs:
            (user1, user2), weight = pair_info
            # Ensure nodes exist (should be guaranteed by prior loop)
            if graph.has_node(user1) and graph.has_node(user2):
                graph.add_edge(user1, user2, weight=round(weight, 3))

        if not graph.edges():
            tui.tui_print_info(
                f"Correlation graph built, but no edges found "
                f"with threshold {abs(current_threshold):.2f}."
            )
        else:
            tui.tui_print_info(
                f"Correlation graph built with {graph.number_of_nodes()} nodes "
                f"and {graph.number_of_edges()} edges "
                f"(threshold {abs(current_threshold):.2f})."
            )

        # Community detection
        if graph.number_of_edges() > 0:
            try:
                partition: Dict[str, int] = community_louvain.best_partition(
                    graph, weight="weight"
                )
                nx.set_node_attributes(graph, partition, "community")
                num_communities: int = len(set(partition.values()))
                tui.tui_print_info(
                    f"Community detection applied. Found {num_communities} communities."
                )
            except Exception as e:
                tui.tui_print_error(f"Error during community detection: {e}")
        else:
            tui.tui_print_info(
                "Skipping community detection as there are no edges in the graph."
            )
            default_partition: Dict[str, int] = {
                node: 0 for node in graph.nodes()
            }
            nx.set_node_attributes(graph, default_partition, "community")

        return graph


    def get_communities(
        self, graph: nx.Graph | None = None
    ) -> Dict[int, List[str]]:
        """
        Extracts communities from a graph.

        The graph should have a \'community\' attribute for each node, typically
        set by `build_correlation_graph`.

        Args:
            graph: A NetworkX graph. If None, `build_correlation_graph` is called
                   internally using the instance\'s default threshold.

        Returns:
            A dictionary where keys are community IDs (int) and values are lists
            of usernames (str) in that community. Sorted by community size
            (descending) and then by community ID.
        """
        target_graph: nx.Graph
        if graph is None:
            target_graph = self.build_correlation_graph(
                threshold=self.default_correlation_threshold
            )
        else:
            target_graph = graph

        if not target_graph or not target_graph.nodes():
            tui.tui_print_warning(
                "Graph is empty or has no nodes. Cannot extract communities."
            )
            return {}

        communities: Dict[int, List[str]] = defaultdict(list)
        for node, data in target_graph.nodes(data=True):
            community_id = data.get("community")
            if community_id is not None:
                communities[community_id].append(node)
            else:
                # Should not happen if graph built by this class
                communities[-1].append(node)  # Unclassified

        # Sort communities by size (desc) then ID (asc for tie-breaking)
        sorted_communities_list = sorted(
            communities.items(),
            key=lambda item: (len(item[1]), item[0]),
            reverse=True,
        )
        return dict(sorted_communities_list)


    def get_users_sorted_by_correlation(
        self, top_n: int | None = None
    ) -> List[str]:
        """
        Sorts users by their average absolute correlation with other users.

        Users with no correlation data or only self-correlation will be at the
        end or have a mean correlation of 0.

        Args:
            top_n: If provided, return only the top N users.

        Returns:
            A list of usernames sorted by their mean absolute correlation
            in descending order.
        """
        if self.correlation_matrix.empty or len(self.correlation_matrix.columns) < 2:
            tui.tui_print_info(
                "Correlation matrix is empty or has less than 2 users. "
                "Cannot sort by correlation."
            )
            return self.user_list[:top_n] if top_n else self.user_list

        mean_abs_correlations: Dict[str, float] = {}
        for user in self.correlation_matrix.columns:
            corrs_for_user: pd.Series = self.correlation_matrix[user].drop(user)
            abs_corrs: pd.Series = corrs_for_user.abs()
            if abs_corrs.notna().any():
                mean_abs_correlations[user] = abs_corrs.mean(skipna=True)
            else:
                mean_abs_correlations[user] = 0.0

        sorted_users: List[str] = sorted(
            mean_abs_correlations.keys(),
            key=lambda u: mean_abs_correlations[u],
            reverse=True,
        )

        tui.tui_print_info(
            f"Users sorted by mean absolute correlation: "
            f"{sorted_users[:10]}... (showing top 10 if many)"
        )

        return sorted_users[:top_n] if top_n else sorted_users


    def get_activity_intervals(
        self, user_list: List[str] | None = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generates activity intervals (online/offline periods) for users.

        The output format is suitable for `plotly.figure_factory.create_gantt`.

        Args:
            user_list: Optional list of users. If None, uses all users from
                       `self.user_list`.

        Returns:
            A dictionary where keys are usernames and values are lists of dicts,
            each representing an interval:
            {\'Task\': username, \'Start\': start_time_dt,
             \'Finish\': end_time_dt, \'Resource\': \'Online\' or \'Offline\'}
        """
        if self.df_resampled.empty:
            tui.tui_print_warning(
                "Resampled DataFrame is empty. Cannot generate activity intervals."
            )
            return {}

        target_users: List[str] = user_list if user_list else self.user_list
        if not target_users:
            tui.tui_print_warning(
                "No target users specified for activity intervals."
            )
            return {}

        all_user_intervals: Dict[str, List[Dict[str, Any]]] = {}
        # Ensure freq is available, default to 1 minute if not.
        # Use cached frequency seconds
        if self.resample_seconds is None:
            self._infer_resample_offset_and_seconds()
        resample_freq_seconds: float = float(self.resample_seconds or 60.0)

        for user in target_users:
            if user not in self.df_resampled.columns:
                tui.tui_print_warning(
                    f"User {user} not found in resampled data. "
                    "Skipping for interval generation."
                )
                continue

            user_activity: pd.Series = self.df_resampled[user]
            user_intervals: List[Dict[str, Any]] = []

            if user_activity.empty:
                all_user_intervals[user] = []
                continue

            current_status_str: str | None = None
            start_time_dt: pd.Timestamp | None = None

            for i in range(len(user_activity)):
                timestamp: pd.Timestamp = user_activity.index[i]
                status_is_online: bool = user_activity.iloc[i] > 0

                period_status_str: str = "Online" if status_is_online else "Offline"

                if current_status_str is None:
                    current_status_str = period_status_str
                    start_time_dt = timestamp
                elif current_status_str != period_status_str:
                    # Status changed, record previous interval
                    # End time is the start of the current differing one
                    end_time_dt: pd.Timestamp = timestamp
                    if start_time_dt is not None: # Ensure start_time_dt was set
                        user_intervals.append(
                            {
                                "Task": user,
                                "Start": start_time_dt.to_pydatetime(),
                                "Finish": end_time_dt.to_pydatetime(),
                                "Resource": current_status_str,
                            }
                        )
                    current_status_str = period_status_str
                    start_time_dt = timestamp
            
            # Record the last interval
            if current_status_str is not None and start_time_dt is not None:
                # End of last interval extends by one resample period duration
                last_interval_end_time: pd.Timestamp = (
                    user_activity.index[-1]
                    + pd.to_timedelta(resample_freq_seconds, unit="s")
                )
                user_intervals.append(
                    {
                        "Task": user,
                        "Start": start_time_dt.to_pydatetime(),
                        "Finish": last_interval_end_time.to_pydatetime(),
                        "Resource": current_status_str,
                    }
                )
            all_user_intervals[user] = user_intervals
        return all_user_intervals


    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Provides summary statistics about the analyzed data.

        Returns:
            A dictionary containing:
                - num_users (int): Number of users.
                - num_resampled_periods (int): Number of resampled time periods.
                - avg_online_periods_per_user (float): Average number of online
                  periods per user.
                - total_duration_analyzed (str): Human-readable total duration.
        """
        if self.df_resampled.empty:
            return {
                "num_users": 0,
                "num_resampled_periods": 0,
                "avg_online_periods_per_user": 0.0,
                "total_duration_analyzed": "N/A",
            }

        num_users: int = len(self.user_list)
        num_periods: int = len(self.df_resampled)
        avg_online_val: float = self.df_resampled.sum(axis=0).mean()
        avg_online_periods: float = round(avg_online_val, 2) if not pd.isna(avg_online_val) else 0.0


        total_duration_str: str = "N/A"
        # Use cached frequency for calculations
        if num_periods > 0 and self.resample_offset is not None:
            freq_timedelta = pd.to_timedelta(self.resample_offset)
            total_seconds: float
            if num_periods == 1:
                total_seconds = freq_timedelta.total_seconds()
            else:  # num_periods > 1
                total_seconds = (
                    self.df_resampled.index[-1] - self.df_resampled.index[0]
                ).total_seconds() + freq_timedelta.total_seconds()

            days: int = int(total_seconds // (24 * 3600))
            hours: int = int((total_seconds % (24 * 3600)) // 3600)
            minutes: int = int((total_seconds % 3600) // 60)

            freq_str_repr = (
                self.df_resampled.index.freqstr
                if self.df_resampled.index.freqstr
                else "unknown_freq"
            )
            if num_periods == 1:
                total_duration_str = (
                    f"{days}d {hours}h {minutes}m "
                    f"(single period: {freq_str_repr})"
                )
            else:
                total_duration_str = f"{days}d {hours}h {minutes}m"

        elif num_periods > 0:  # Freq unknown but data exists
            min_ts = self.df_resampled.index.min().strftime("%Y-%m-%d %H:%M")
            max_ts = self.df_resampled.index.max().strftime("%Y-%m-%d %H:%M")
            total_duration_str = (
                f"Approx. from {min_ts} to {max_ts} (freq unknown)"
            )

        summary: Dict[str, Any] = {
            "num_users": num_users,
            "num_resampled_periods": num_periods,
            "avg_online_periods_per_user": avg_online_periods,
            "total_duration_analyzed": total_duration_str,
        }
        tui.tui_print_info(f"Summary Statistics: {summary}")
        return summary


    def _calculate_jaccard_indices(self) -> pd.DataFrame:
        """
        Calculates Jaccard index matrix between users' online sessions.
        
        Returns:
            A pandas DataFrame representing the Jaccard index matrix.
        """
        if self.df_resampled.empty or len(self.df_resampled.columns) < 2:
            tui.tui_print_warning(
                "Not enough data or users to calculate Jaccard indices."
            )
            return pd.DataFrame()

        # Convert to binary (0/1) for Jaccard calculation
        df_binary: pd.DataFrame = self.df_resampled.astype(bool).astype(int)

        users: pd.Index = df_binary.columns

        # Vectorized pairwise intersection using matrix multiplication
        X: np.ndarray = df_binary.to_numpy(dtype=np.int32, copy=False)
        intersections: np.ndarray = X.T @ X  # shape (U, U)

        # Column sums (online counts per user)
        sums: np.ndarray = X.sum(axis=0, dtype=np.int64)  # shape (U,)
        unions: np.ndarray = sums[:, None] + sums[None, :] - intersections

        # Avoid division by zero; where union==0, set jaccard to 0
        with np.errstate(divide="ignore", invalid="ignore"):
            jaccard_arr: np.ndarray = np.where(
                unions > 0, intersections / unions, 0.0
            )

        # Ensure diagonal is exactly 1.0 when a user has any online activity, else 0.0
        diag_values = np.where(sums > 0, 1.0, 0.0)
        np.fill_diagonal(jaccard_arr, diag_values)

        jaccard_df = pd.DataFrame(jaccard_arr, index=users, columns=users, dtype=float)
        tui.tui_print_info("Jaccard index matrix calculated (vectorized).")
        return jaccard_df


    def get_jaccard_matrix(self) -> pd.DataFrame:
        """
        Retrieves the calculated Jaccard index matrix.

        Returns:
            A pandas DataFrame with user-to-user Jaccard indices.
        """
        return self.jaccard_matrix


    def get_significant_jaccard_pairs(
        self, threshold: float | None = None
    ) -> List[Tuple[Tuple[str, str], float]]:
        """
        Identifies pairs of users with Jaccard index above a given threshold.

        Args:
            threshold: The Jaccard index threshold. If None, the
                       instance's default_jaccard_threshold is used.

        Returns:
            A list of tuples, where each tuple contains:
            ((user1, user2), jaccard_index).
            The list is sorted by the Jaccard index in descending order.
        """
        current_threshold: float = (
            threshold
            if threshold is not None
            else self.default_jaccard_threshold
        )

        if self.jaccard_matrix.empty:
            return []

        significant_pairs: List[Tuple[Tuple[str, str], float]] = []
        columns: List[str] = self.jaccard_matrix.columns.tolist()

        # Iterate over the upper triangle of the matrix
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                user1: str = columns[i]
                user2: str = columns[j]
                jaccard_value: float = self.jaccard_matrix.iloc[i, j]

                if jaccard_value >= current_threshold:
                    significant_pairs.append(((user1, user2), jaccard_value))

        significant_pairs.sort(key=lambda x: x[1], reverse=True)
        tui.tui_print_info(
            f"Found {len(significant_pairs)} significant Jaccard pairs "
            f"with threshold >= {current_threshold:.2f}"
        )
        return significant_pairs



# --- Helper Functions ---

def create_activity_gantt_chart(
    activity_intervals: Dict[str, List[Dict[str, Any]]],
    title: str = "User Activity Gantt Chart",
) -> Union[Any, None]: # Plotly figure type is not standard, use Any
    """
    Generates a Plotly Gantt chart from activity intervals.

    Requires Plotly to be installed.

    Args:
        activity_intervals: A dictionary of activity intervals, typically from
                            `TemporalAnalyzer.get_activity_intervals()`.
        title: The title of the Gantt chart.

    Returns:
        A Plotly Figure object if successful, otherwise None.
    """
    try:
        import plotly.figure_factory as ff
        import plotly.io as pio # For potential saving, not used directly here for display
        # pio.templates.default = "plotly_dark" # Optional: theme
    except ImportError:
        tui.tui_print_warning(
            "Plotly is not installed. Cannot generate Gantt chart. "
            "Please install with `pip install plotly`."
        )
        return None

    df_tasks: List[Dict[str, Any]] = []
    for user_intervals in activity_intervals.values():
        df_tasks.extend(user_intervals)

    if not df_tasks:
        tui.tui_print_info("No data available to generate Gantt chart.")
        return None

    colors: Dict[str, str] = {
        "Online": "rgb(0, 200, 0)",  # Green
        "Offline": "rgb(220, 0, 0)", # Red
    }

    try:
        fig = ff.create_gantt(
            df_tasks,
            colors=colors,
            index_col="Resource", # \'Online\' or \'Offline\'
            show_colorbar=True,
            group_tasks=True,    # Groups by \'Task\' (username)
            title=title,
        )
        fig.update_layout(xaxis_title="Time", yaxis_title="User")
        tui.tui_print_info(
            f"Gantt chart \'{title}\' created. "
            "Display with fig.show() or save externally."
        )
        # Example save:
        # chart_filename = title.lower().replace(' ', '_') + ".html"
        # pio.write_html(fig, file=chart_filename, auto_open=False)
        # tui.tui_print_info(f"Gantt chart saved to {chart_filename}")
        return fig
    except Exception as e:
        tui.tui_print_error(f"Error creating Gantt chart: {e}")
        return None



# --- Main Execution Block (for testing and demonstration) ---

if __name__ == "__main__":
    tui.tui_print_highlight("Testing TemporalAnalyzer module...")

    # Dummy data setup
    date_rng = pd.date_range(
        start="2023-01-01 00:00",
        end="2023-01-01 01:00",
        freq="1min",
        tz="UTC",
    )
    data_payload = {
        "Alice": [
            1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1,
            1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1,
            0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1,
        ],
        "Bob": [
            0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1,
            0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1,
            1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1,
        ],
        "Charlie": [
            1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0,
            1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0,
            0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1,
        ],
    }
    df_activity_test = pd.DataFrame(data_payload, index=date_rng).astype(bool)

    tui.tui_print_info("Initializing TemporalAnalyzer with dummy data...")
    analyzer = TemporalAnalyzer(
        df_activity_test, resample_period="2T", correlation_threshold=0.3
    )

    tui.tui_print_info("\\n--- Correlation Matrix ---")
    corr_matrix_test = analyzer.get_correlation_matrix()
    if not corr_matrix_test.empty:
        tui.tui_print_code(
            corr_matrix_test.to_string(), language="text", title="Correlation Matrix"
        )
    else:
        tui.tui_print_warning("Correlation matrix is empty.")

    tui.tui_print_info("\\n--- Significant Pairs (threshold from init) ---")
    sig_pairs_default_test = analyzer.get_significant_pairs()
    if sig_pairs_default_test:
        for pair, corr_val in sig_pairs_default_test:
            tui.tui_print_info(f"  {pair}: {corr_val:.3f}")
    else:
        tui.tui_print_warning("No significant pairs found with default threshold.")

    tui.tui_print_info("\\n--- Significant Pairs (threshold override 0.1) ---")
    sig_pairs_override_test = analyzer.get_significant_pairs(threshold=0.1)
    if sig_pairs_override_test:
        for pair, corr_val in sig_pairs_override_test:
            tui.tui_print_info(f"  {pair}: {corr_val:.3f}")
    else:
        tui.tui_print_warning(
            "No significant pairs found with overridden threshold 0.1."
        )

    tui.tui_print_info("\\n--- Building Correlation Graph (threshold from init) ---")
    graph_test = analyzer.build_correlation_graph()
    if graph_test.nodes():
        tui.tui_print_info(
            f"  Graph nodes: {graph_test.number_of_nodes()}, "
            f"edges: {graph_test.number_of_edges()}"
        )
        if graph_test.edges():
            tui.tui_print_info("  Communities:")
            communities_test = analyzer.get_communities(graph=graph_test)
            for comm_id, users_in_comm in communities_test.items():
                tui.tui_print_info(f"    Community {comm_id}: {users_in_comm}")
    else:
        tui.tui_print_warning("Graph is empty.")

    tui.tui_print_info("\\n--- Users Sorted by Correlation ---")
    sorted_users_test = analyzer.get_users_sorted_by_correlation(top_n=5)
    tui.tui_print_info(f"  Top 5 correlated users: {sorted_users_test}")

    tui.tui_print_info("\\n--- Activity Intervals (for Gantt) ---")
    activity_intervals_data_test = analyzer.get_activity_intervals()
    if "Alice" in activity_intervals_data_test and activity_intervals_data_test["Alice"]:
        tui.tui_print_info("Sample intervals for Alice:")
        for interval in activity_intervals_data_test["Alice"][:3]:
            start_str = interval["Start"].strftime("%Y-%m-%d %H:%M:%S")
            finish_str = interval["Finish"].strftime("%Y-%m-%d %H:%M:%S")
            tui.tui_print_info(
                f"  Start: {start_str}, Finish: {finish_str}, "
                f"Resource: {interval['Resource']}"
            )
        # gantt_fig_test = create_activity_gantt_chart(
        #     activity_intervals_data_test, title="Test User Activity"
        # )
        # if gantt_fig_test:
        #     pass # fig.show() or save
    else:
        tui.tui_print_warning(
            "No activity intervals found for Alice or data is empty."
        )

    tui.tui_print_info("\\n--- Summary Stats ---")
    summary_test = analyzer.get_summary_stats()
    for key, value in summary_test.items():
        tui.tui_print_info(f"  {key.replace('_',' ').capitalize()}: {value}")

    tui.tui_print_success("\\nTemporalAnalyzer tests completed.") 