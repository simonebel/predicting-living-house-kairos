from plotly.subplots import make_subplots
import plotly.graph_objects as go


def chart_plot(
    graph, dataframe, x, y, plot_args, color=None, ranges=None, img_name=None
):
    """
    This function is used to display any type of chart available within plotly.

    Args:
        graph (plotly.express._chart_types):
            The plotly function that will be used for the plot.

        dataframe (pandas.Dataframe):
            The dataframe sotring the data to display.

        x (list or numpy.array or pandas.Series):
            The data to display along the x axis

        y (list or numpy.array or pandas.Series):
            The data to display along the y axis

        plot_args (dict):
            A dictionnary storing the arguments for the plot. The title, the width, the height...

        color (pandas.Series, optional, default=None):
            The color to apply to each point that will be displayed.

        ranges (Tuple(List(int)), optional, default=None):
            Precise a range along the x axis and y axis to display the chart.
            The first arg of the tuple define the range for the x axis.
            The second arg of the tuple define the range for the y axis.

        img_name (string, optional, default=None):
            The name of the image file that will be save.

    """

    # Create the figure
    fig = graph(dataframe, x=x, y=y, color=color)

    # Apply a range to the figure, if provided
    if ranges is not None:
        fig.update_xaxes(range=ranges[0])
        fig.update_yaxes(range=ranges[1])

    # If an image name is provided, save the figure in png
    if img_name:

        # remove the title for the report
        title = plot_args.pop("title")
        title_x = plot_args.pop("title_x")

        # update the layout with the title
        fig.update_layout(plot_args)

        # our image folder path
        img_path = "static/img/"

        # save the figure
        fig.write_image(img_path + img_name)

        # Re store the title for the plot
        plot_args["title"] = title
        plot_args["title_x"] = title_x

    # Update the layout with the title
    fig.update_layout(plot_args)

    # Plot our figure
    fig.show()


def chart_subplot(graph, dataframe, generate_data, n_columns, plot_args, img_name=None):
    """
    This function is used to display a subplot of any type of chart available within plotly.

    Args:
        graph (plotly.express._chart_types):
            The plotly function that will be used for the plot.

        dataframe (pandas.Dataframe):
            The dataframe sotring the data to display.

        generate_data (function):
            The function that will returned the data along the x and y axis.
            The function take as input a  dataframe (pandas.Dataframe) and a unique_id (string)
            and return a tuple x,y

        n_columns (int):
            The number of columns of the grid.

        plot_args (dict):
            A dictionnary storing the arguments for the plot. The title, the width, the height...

        img_name (string, optiona, default = None):
            The name of the image file that will be save. If None the plot is not saved.
    """

    row, col = 1, 1

    # Compute the number of row given the number of user
    rows = (
        len(dataframe.user_id.unique()) // n_columns
        if len(dataframe.user_id.unique()) % n_columns == 0
        else len(dataframe.user_id.unique()) // n_columns + 1
    )

    # Create the subplot figure
    fig = make_subplots(
        rows=rows,
        cols=n_columns,
        x_title=None,
        y_title=None,
        subplot_titles=(
            ["({})".format(k) for k in range(1, len(dataframe.user_id.unique()) + 1)]
        ),
    )

    # Setup x and y axes titles
    x_title = plot_args.pop("x_title")
    y_title = plot_args.pop("y_title")

    # Create a subplot for each user
    for unique_id in sorted(dataframe.user_id.unique()):

        # mod the column position
        if col > n_columns:
            col = 1
            row += 1

        # A custom function to generate x and y data
        x, y = generate_data(dataframe, unique_id)

        # Set up the graph
        graph_args = {"x": x, "y": y, "name": "user " + unique_id[:4]}

        # In case of scatter plot we want markers
        if isinstance(graph(), go.Scatter):
            graph_args["mode"] = "markers"

        # Add the graph to figure
        fig.add_trace(graph(**graph_args), row=row, col=col)

        # Update the x and y axes title of the graph
        fig.update_xaxes(title_text=x_title, row=row, col=col)
        fig.update_yaxes(title_text=y_title, row=row, col=col)

        # Increment the column index
        col += 1

    # If an image name is provided, save the figure in png
    if img_name:
        # remove the title for the report
        title = plot_args.pop("title_text")
        title_x = plot_args.pop("title_x")

        # update the layout with the title
        fig.update_layout(plot_args)

        # our image folder path
        img_path = "static/img/"

        # save the figure
        fig.write_image(img_path + img_name)

        # Re store the title for the plot
        plot_args["title_text"] = title
        plot_args["title_x"] = title_x

    # Update the layout with the title
    fig.update_layout(plot_args)

    # Plot our figure
    fig.show()


def visualize_filtering(
    dataframe, filtered_dataframe, plot_args, n_users=2, img_name=None
):
    """
    This function is used to visualize the effect of filtering on the spatial data.

    Args:
        dataframe (pandas.Dataframe):
            The original dataframe

        filtered_dataframe (pandas.Dataframe):
            The filtered dataframe.

        plot_args (dict):
            A dictionnary storing the arguments for the plot. The title, the width, the height...

        n_users (int):
            The number of user to display.

        img_name (string, optional, default = None):
            The name of the image file that will be save. If None the plot is not saved.
    """

    # Utility argument to number and name the subplots
    letter = ["a", "b"]
    state = ["Orginal", "Filtered"]
    num = []
    for k in range(1, len(dataframe.user_id.unique()[:n_users]) + 1):
        n = [k] * 2
        num.extend(n)

    # Create the subplot figure
    fig = make_subplots(
        rows=n_users,
        cols=2,
        subplot_titles=(
            [
                "{} ({}.{})".format(
                    state[0] if k % 2 != 0 else state[1],
                    num[k - 1],
                    letter[0] if k % 2 != 0 else letter[1],
                )
                for k in range(1, len(num) + 1)
            ]
        ),
    )

    # Create the sub-figure for each user
    row = 1
    for id in dataframe.user_id.unique()[:n_users]:
        col = 1

        # We get the data of a specific user
        user_df = dataframe[dataframe.user_id == id]
        filtered_user_df = filtered_dataframe[filtered_dataframe.user_id == id]

        # Add the plot of the original data
        fig.add_trace(
            go.Scatter(mode="markers", x=user_df.lon, y=user_df.lat, name=id[:4]),
            row=row,
            col=col,
        )

        # Add the plot of the filtered data
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=filtered_user_df.lon,
                y=filtered_user_df.lat,
                name=id[:4],
            ),
            row=row,
            col=col + 1,
        )

        # Define x and y axes titles
        x_title = "Longitude"
        y_title = "Latitude"

        # Update the x and y axes titles of the original subplot
        fig.update_xaxes(title_text=x_title, row=row, col=col)
        fig.update_yaxes(title_text=y_title, row=row, col=col)

        # Update the x and y axes titles of the filtered subplot
        fig.update_xaxes(title_text=x_title, row=row, col=col + 1)
        fig.update_yaxes(title_text=y_title, row=row, col=col + 1)

        # Increment the row
        row += 1

    # If an image name is provided, save the figure in png
    if img_name:

        # Remove title from the figure to use it in the report
        title = plot_args.pop("title")
        title_x = plot_args.pop("title_x")

        # Our path to images folder
        img_path = "static/img/"

        # Update the layout without the title
        fig.update_layout(plot_args)

        # Save our image
        fig.write_image(img_path + img_name)

        # Store back the title in the plotting arguments
        plot_args["title"] = title
        plot_args["title_x"] = title_x

    # Re update our layout with title
    fig.update_layout(plot_args)

    # Plot our figure
    fig.show()


def single_map_view(dataframe, plot_args, img_name=None):
    """
    This function is used to display the gps coordinate on a map for a single user.

    Args:
        dataframe (pandas.Dataframe):
            The dataframe containing the data for one user

        plot_args (dict):
            A dictionnary storing the arguments for the plot. The title, the width, the height...

        n_users (int):
            The number of user to display.

        img_name (string, optional, default = None):
            The name of the image file that will be save. If None the plot is not saved.
    """

    # Create out scatter map
    fig = go.Figure(
        go.Scattermapbox(
            lat=dataframe.lat,
            lon=dataframe.lon,
            mode="markers",
        )
    )

    # If an image name is provided, save the figure in png
    if img_name:

        # Remove title from the figure to use it in the report
        title = plot_args.pop("title")
        title_x = plot_args.pop("title_x")

        # Our path to images folder
        img_path = "static/img/"

        # Update the layout without the title
        fig.update_layout(plot_args)

        # Save our image
        fig.write_image(img_path + img_name)

        # Store back the title in the plotting arguments
        plot_args["title"] = title
        plot_args["title_x"] = title_x

    # Re update our layout with title
    fig.update_layout(plot_args)

    # Plot our figure
    fig.show()


def map_view(dataframe, img_name=None):
    """
    This function is used to display the gps coordinate on a map for all the users.
    This function is different of the different subplot as subplotting mapbox isn't support now in plotly.

    Args:
        dataframe (pandas.Dataframe):
            The dataframe containing the data of all the  users

        img_name (string, optional, default = None):
            The name of the image file that will be save. If None the plot is not saved.
    """

    data = []

    # Setup the layout
    layout = dict(
        title="Map view of GPS coordinate measure for each users",
        autosize=True,
        hovermode="closest",
    )

    # For each user, create a scatter mapbox
    for idx, unique_id in enumerate(dataframe.user_id.unique()):

        # Get the data of a user
        user_df = dataframe[dataframe.user_id == unique_id]
        user_df = user_df.sort_values("timestamp_client")

        # Create a mapbox id
        subplot = "mapbox" + str(idx + 1)

        # Append to data the graph of one user
        data.append(
            go.Scattermapbox(
                lat=user_df.lat,
                lon=user_df.lon,
                subplot=subplot,
                mode="markers",
                name=unique_id[:4],
            )
        )

        # Add to the layout the layout of the graph
        layout[subplot] = dict(
            domain=dict(x=[], y=[]),
            center=dict(lat=user_df.lat.median(), lon=user_df.lon.median()),
            style="open-street-map",
            zoom=5,
        )

    # For each graph create, we will set them to a pos in the layout
    z = 0
    col = 3
    row = 4
    for y in reversed(range(row)):
        for x in range(col):

            # Get a graph id
            subplot = subplot = "mapbox" + str(z + 1)

            # Set the x domain (x pos) in the subplot
            layout[subplot]["domain"]["x"] = [
                float(x + 0.05) / float(col),
                float(x + 1) / float(col),
            ]

            # Set the y domain (y pos) in the subplot
            layout[subplot]["domain"]["y"] = [
                float(y + 0.05) / float(row),
                float(y + 1) / float(row),
            ]

            z = z + 1
            if z > 12:
                break

    # If an image name is provided, save the figure in png
    if img_name:

        # Remove the tilte from the layout
        title = layout.pop("title")

        # Create the figure with graphs and layouts
        fig = go.FigureWidget(data=data, layout=layout)

        # Our path to images folder
        img_path = "static/img/"

        # Update the layout without the title
        fig.update_layout(
            height=1000,
            width=1400,
        )

        # Save the image
        fig.write_image(img_path + img_name)

    # Update the layout with the title
    fig.update_layout(height=1000, width=1400, title=title, title_x=0.5)

    # Plot the figure
    fig.show()


def map_view_prediction(dataframe, prediction, img_name=None):
    """
    This function is used to display the gps coordinate and the living location prediction on a map for all the users.

    Args:
        dataframe (pandas.Dataframe):
            The dataframe containing the data of all the  users

        prediction (dict):
            A dictionnary containing the user id (key) and the living location prediction (value)

        img_name (string, optional, default = None):
            The name of the image file that will be save. If None the plot is not saved.
    """

    data = []

    # Setup the layout
    layout = dict(
        title="Map view of predicting location for each users",
        autosize=True,
        hovermode="closest",
    )

    # For each user, create a scatter mapbox with the GPS coordinates and the location prediction
    for idx, unique_id in enumerate(sorted(dataframe.user_id.unique())):

        # Get the data of a user
        user_df = dataframe[dataframe.user_id == unique_id]
        user_df = user_df.sort_values("timestamp_client")

        # Append to the data the predicted location
        lat = user_df.lat.to_list()
        lat.append(prediction[unique_id][0])
        lon = user_df.lon.to_list()
        lon.append(prediction[unique_id][1])

        # GPS coordinates will be display in blue
        color = ["blue" for k in range(len(user_df.lat))]
        # Predicted location in red
        color.append("red")

        # The predicted location will be bigger
        size = [5 for k in range(len(user_df.lat))]
        size.append(10)

        # Create a mapbox id
        subplot = "mapbox" + str(idx + 1)

        # Append to data the graph of one user
        data.append(
            go.Scattermapbox(
                lat=lat,
                lon=lon,
                subplot=subplot,
                mode="markers",
                name=unique_id[:4],
                marker={"color": color, "size": size, "opacity": 0.6},
            )
        )
        # Add to the layout the layout of the graph
        layout[subplot] = dict(
            domain=dict(x=[], y=[]),
            center=dict(lat=prediction[unique_id][0], lon=prediction[unique_id][1]),
            style="open-street-map",
            zoom=17,
        )

    # For each graph create, we will set them to a pos in the layout
    z = 0
    col = 3
    row = 4
    for y in reversed(range(row)):
        for x in range(col):
            # Get a graph id
            subplot = subplot = "mapbox" + str(z + 1)

            # Set the x domain (x pos) in the subplot
            layout[subplot]["domain"]["x"] = [
                float(x + 0.05) / float(col),
                float(x + 1) / float(col),
            ]

            # Set the y domain (y pos) in the subplot
            layout[subplot]["domain"]["y"] = [
                float(y + 0.05) / float(row),
                float(y + 1) / float(row),
            ]
            z = z + 1
            if z + 1 > len(dataframe.user_id.unique()):
                break

    # Create the figure with graphs and layouts
    fig = go.FigureWidget(data=data, layout=layout)

    # If an image name is provided, save the figure in png
    if img_name:

        # Remove the tilte from the layout
        title = layout.pop("title")

        # Our path to images folder
        img_path = "static/img/"

        # Update the layout without the title
        fig.update_layout(
            height=1000,
            width=1400,
        )

        # Save the image
        fig.write_image(img_path + img_name)

        # Update the layout with the title
        fig.update_layout(title=title, title_x=0.5)

    # Update the size of layout
    fig.update_layout(height=1000, width=1400)

    # Plot the figure
    fig.show()
