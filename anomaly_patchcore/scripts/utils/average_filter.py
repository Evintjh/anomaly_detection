import numpy as np


class AverageFilter:
    def __init__(self, filter_size=5):
        """
        Initialize the AverageFilter with a specific filter size.
        Args:
            filter_size (int): Number of frames to consider for averaging.
        """
        self.filter_size = filter_size
        self.moving_avg_filter = []

    def moving_avg(self, anomaly_map: np.ndarray) -> np.ndarray:
        """
        Apply a moving average filter to a 2D anomaly map.

        Args:
            anomaly_map (np.ndarray): The input anomaly map to be filtered.

        Returns:
            np.ndarray: The filtered anomaly map.
        """
        # Ensure the input is a NumPy array
        if not isinstance(anomaly_map, np.ndarray):
            raise TypeError("Input anomaly_map must be a NumPy array")

        # Add the new anomaly map to the filter list
        if len(self.moving_avg_filter) == self.filter_size:
            self.moving_avg_filter.pop(0)  # Remove the oldest map if the list is full

        self.moving_avg_filter.append(anomaly_map)

        # Compute the average over stored maps
        return np.mean(self.moving_avg_filter, axis=0)

    def reset(self):
        """
        Reset the filter by clearing the stored anomaly maps.
        """
        self.moving_avg_filter.clear()
