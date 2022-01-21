# Data
import numpy as np
import pandas as pd

# ML
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances
from sklearn.neighbors import NearestNeighbors

# Reverse geocode
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


def generate_time_elapsed(dataframe):
    """
    This function generate the time elapsed between two consecutives mesure.

    Args :
        dataframe (pandas.Dataframe):
            The dataframe containing the GPS coordinates of all the users


    Returns :
         dataframe (pandas.Dataframe):
           The dataframe containing the time elapsed
    """

    # Create a new columns in the dataframe, where each value is the time elapsed between the current measure and the one before.
    dataframe["time_elapsed"] = dataframe.timestamp_client.diff()

    return dataframe


def generate_distance_travelled(dataframe):
    """
    This function generate the distance travelled between two consecutives mesure.

    Args :
        dataframe (pandas.Dataframe):
            The dataframe containing the GPS coordinates of all the users


    Returns :
         dataframe (pandas.Dataframe):
           The dataframe containing the distance travelled
    """

    # Get the coordinates
    gps_coordinate = dataframe[["lat", "lon"]].to_numpy()

    # Convert them to rad for haversine distance
    gps_coordinate_rad = np.radians(gps_coordinate)

    # Compute the haversine distance between the current point and the one before
    distance_travelled = np.array(
        [
            haversine_distances([gps_coordinate_rad[i - 1], gps_coordinate_rad[i]])
            for i in range(1, len(gps_coordinate_rad))
        ]
    )

    # Convert the distance to km
    distance_travelled = distance_travelled[:, 1][:, 0] * 6371

    # Append a zero for the first point
    distance_travelled = np.insert(distance_travelled, 0, 0.0)

    # Store the distance in a new column
    dataframe["distance_travelled"] = distance_travelled

    return dataframe


def filtering_spatial_distance(dataframe, threshold):
    """
    This function filters the data along the distance.
    If the distance between two points is larger than threshold we remove the current point

    Args :
        dataframe (pandas.Dataframe):
            The dataframe containing the GPS coordinates of all the users

        threshold (float):
            The distance for which the value are filtered (km)

    Returns :
         filtering_dataframe (pandas.Dataframe):
           The filtered dataframe
    """

    # Get the distance travelled
    distance = dataframe.distance_travelled.to_numpy()

    # Create the mask to filter the data
    filtering_mask = distance < threshold

    # Get the filtered dataframe
    filtering_dataframe = dataframe[filtering_mask]

    return filtering_dataframe


def filtering_speed(dataframe, threshold):
    """
    This function filters the data along the speed.
    If the speed between two points is larger than threshold we remove the current point

    Args :
        dataframe (pandas.Dataframe):
            The dataframe containing the GPS coordinates of all the users

        threshold (float):
            The speed for which the value are filtered (m/s)

    Returns :
         filtering_dataframe (pandas.Dataframe):
           The filtered dataframe
    """

    # Filter the zero time elapsed  and convert the time to second
    dataframe = dataframe[dataframe.time_elapsed != np.timedelta64(0, "s")]

    # Get the time elapsed and convert it to int
    time_elapsed = dataframe.time_elapsed.to_numpy()[1:]
    time_elapsed = time_elapsed.astype("timedelta64[s]").astype(int)

    # Get the distance travelled and convert the distance to meter
    distance_travelled = dataframe.distance_travelled.to_numpy()[1:]
    distance_travelled = distance_travelled * 1000

    # Compute the speed
    speed = distance_travelled / time_elapsed

    # Filter the data
    speed_mask = speed < threshold
    speed_mask = np.insert(speed_mask, 0, True)

    return dataframe[speed_mask]


def filtering_df(dataframe, thrsh_speed, thrsh_distance):
    """
    This function apply all the filtering strategies (distance and speed).

    Args :
        dataframe (pandas.Dataframe):
            The dataframe containing the GPS coordinates of all the users

        thrsh_speed (float):
            The speed for which the value are filtered

        thrsh_distance (float):
            The distance for which the value are filtered

    Returns :
         filtering_dataframe (pandas.Dataframe):
           The filtered dataframe
    """

    # Compute the time elapsed and distance travelled
    dataframe = generate_time_elapsed(dataframe)
    dataframe = generate_distance_travelled(dataframe)

    # Get the filtered dataframe
    dataframe = filtering_spatial_distance(dataframe, thrsh_distance)
    dataframe = filtering_speed(dataframe, thrsh_speed)

    return dataframe


def compute_distance_from_point(from_point, to_points):
    """
    This function apply all the filtering strategies (distance and speed).

    Args :
        from_point (np.array or list):
            A list of points from which we want to compute the distance

        to_points (np.array or list):
            A list of points to which we want to compute the distance

    Returns :
         distance_from_point (np.array):
           A list (or a matrix) containing the distance bewteen from_point and to_points
    """

    # Convert the coordinate to rad
    from_point = np.radians(from_point)
    to_points = np.radians(to_points)

    # Compute the haversine distance and convert it to km
    distance_from_point = np.array(haversine_distances(from_point, to_points))
    distance_from_point = distance_from_point * 6371

    return distance_from_point


def get_center_dist(gps_cluster):
    """
    This function find the point in a cluster with the more nearest neighbors.
    This point will be used as the living prediction

    Args :
        gps_cluster (pandas.Dataframe):
            A dataframe containing the points of a same cluster

    Returns :
         gps_cluster_coord[center_idx] (np.array):
           The gps coordinate of the 'center' of the cluster
    """

    # Get the data from a cluster
    gps_cluster_coord = gps_cluster[["lat", "lon"]].to_numpy()

    # Compute the distance matrix
    distance_from_point = compute_distance_from_point(
        gps_cluster_coord, gps_cluster_coord
    )

    # Get the index of the point with more nearest neighbors
    dist_matrix = pd.DataFrame(distance_from_point)
    center_idx = dist_matrix.sum(axis=1).idxmin()

    return gps_cluster_coord[center_idx]


def get_center_mean(gps_cluster):
    """
    This function find the center of  a cluster.
    This point will be used as the living prediction

    Args :
        gps_cluster (pandas.Dataframe):
            A dataframe containing the points of a same cluster

    Returns :
         location_prediction (np.array):
           The gps coordinate of the center of the cluster
    """

    # Compute the center of a cluster
    location_prediction = gps_cluster[["lat", "lon"]].to_numpy().mean(axis=0)

    return location_prediction


def compute_k_avg_distance(dataframe, n_neighbors):
    """
    This function compute the average distance of k-th neighbors

    Args :
        dataframe (pandas.Dataframe):
            A dataframe containing our data

        n_neighbors (int):
            The number of neighbors to consider

    Returns :
        distances (np.array):
            The sorted distances
    """

    # KNN algorithm
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric="haversine")

    # Calculate the average distance between each point in the data set and its 3 nearest neighbors
    gps_coord = dataframe[["lat", "lon"]].to_numpy()
    neighbors_fit = neighbors.fit(gps_coord)
    distances, indices = neighbors_fit.kneighbors(gps_coord)

    # Sort distance values by ascending
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]

    return distances


def get_living_location(dataframe, dbscan_kwargs, mode):
    """
    This function apply the DBSCAN Alogrithm to all our data and return the living location prediction.

    Args :
        dataframe (pandas.Dataframe):
            A dataframe containing our data

        dbscan_kwargs (dict):
            The argument for DBSCAN

        mode (str, option = 'mean', 'dist') :
            The mode to choose for predicting the living location :
                - mean : return the center of a cluster
                - dist : return the point with the most nearest neighbors

    Returns :
        dataframe (pandas.Dataframe):
            The dataframe containing the data and the label of the cluster

        prediction (dict) :
            A dictionnary containing the user id (key) and the living location prediction (value)
    """

    # Get the coordinates and convert it to rad
    data = dataframe[["lon", "lat"]].to_numpy()
    data = np.radians(data)

    # Set-up the DBSCAN algorithm
    model = DBSCAN(**dbscan_kwargs)

    # Apply DBSCAN
    model.fit(data)

    # Get the label of the cluster
    dataframe["label"] = model.labels_

    # Get the prediction for each user
    prediction = {}
    for user_id in dataframe.user_id.unique():

        # Get the data of a specific user
        user_df = dataframe[dataframe.user_id == user_id]

        # Count the number of points in each cluster
        n_sample_per_cluster = user_df.groupby("label").count()

        # Sorted the cluster by number of sample
        sorted_n_sample_per_cluster = n_sample_per_cluster.sort_values(
            "timestamp_client", ascending=False
        )

        # Get the the denser cluster
        for label in sorted_n_sample_per_cluster.index:

            # If the cluster is the Noise one we take the next cluster
            if label != -1:
                location_label = label
                break

        # We get all the point of the denser cluster
        gps_cluster = user_df[user_df.label == location_label]

        # Apply 'mean' method
        if mode == "mean":
            location_prediction = get_center_mean(gps_cluster)

        # Apply 'dist' method
        if mode == "dist":
            location_prediction = get_center_dist(gps_cluster)

        # Store the prediction
        prediction[user_id] = location_prediction

    return dataframe, prediction


def get_adress_from_coordinates(prediction):
    """
    This function return a location address from GPS coordinates.

    Args :
        prediction (dict):
            A dictionnary containing the user id (key) and the living location prediction (value)

    Returns :
      final_prediction (dict) :
            A dictionnary containing the user id (key),  the living location prediction and the location address (value)
    """

    # Set-up Geopy Agent
    geolocator = Nominatim(user_agent="test")

    # Set-up delay to prevent for being rejected from the API
    reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)

    # Append for each location the associated address
    final_prediction = {}
    for user_id in prediction.keys():

        # Get the location coordinate
        pred = prediction[user_id]

        # convert the coordinate to string
        loc_str = "{}, {}".format(pred[0], pred[1])

        # Reverse the location
        location = reverse(loc_str)

        # Store the address
        final_prediction[user_id] = {"coordinate": pred, "adress": location.address}

    return final_prediction
