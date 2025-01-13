# Contains class and methods for data analysis of dataset
import os
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from torchvision.io import decode_image
from torchvision.utils import make_grid

from bdd_utils import file_utils


class BaseEDA:
    def __init__(self, config: Union[str, Dict[str, Any]]):
        """
        Base EDA class for initializing configuration.
        Args:
            config: A path to a YAML file or a dictionary with configuration.
        """
        if isinstance(config, str):
            self.config = file_utils.load_config(config)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise ValueError("Config must be a YAML file path or a dictionary.")
        print(self.config)
        self.train_labels = self.config.get("train_label")
        self.val_labels = self.config.get("val_label")
        self.train_images = self.config.get("train_images")
        self.val_images = self.config.get("val_images")
        self.train_csv = self.config.get("train_csv")
        self.val_csv = self.config.get("val_csv")

        self._validate_config()

    def _validate_config(self):
        """Validate the required configuration keys."""
        required_keys = ["train_label", "val_label", "train_images", "val_images"]
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"Missing required config key: {key}")

    def load_labels(self, file_path: str) -> Any:
        """Load a JSON label file."""
        return file_utils.read_json(file_path)

    def __repr__(self):
        return f"BaseEDA(config={self.config})"


class BDD100KEDA(BaseEDA):
    def __init__(self, config: Union[str, Dict[str, Any]], output_path: str, num_samples: int = -1):
        """Initialize BDD100EDA class

        Args:
            config (Union[str, Dict[str, Any]]): Path of config file which contains path of train,val images and annotations
            output_path (str): Path to save the output plots
            num_samples (int, optional): Num of samples for data analysis. Defaults to -1.
        """
        super().__init__(config)
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        # Initialize dataframe attributes to None
        self.train_df = None
        self.val_df = None
        self.df_train_val = None
        self._create_dataframe(num_samples)

    def load_train_labels(self) -> Any:
        """Load training labels."""
        return self.load_labels(self.train_labels)

    def load_val_labels(self) -> Any:
        """Load validation labels."""
        return self.load_labels(self.val_labels)

    def _convert2dataframe(self, data: List[Dict]) -> pd.DataFrame:
        """Convert the list of annotation dict to a flattened dataframe
        Args:
            data (List[Dict]): BDD100K annotation list where each item in list corresponds to annotation for single image

        Returns:
            pd.Dataframe: A dataframe which is flattened from the list of dict
        """
        # Initialize lists to store data
        data_flattened = []

        # Iterate over the image and labels
        for item in data:
            image_info = {
                'image_id': item['name'],
                'weather': item['attributes'].get('weather', None),
                'timeofday': item['attributes'].get('timeofday', None),
                'scene': item['attributes'].get('scene', None),
                'timestamp': item['timestamp'],
            }
            if 'labels' not in item:
                obj_info = {
                    'image_id': item['name'],
                    'category': None,
                    'traffic_light_color': None,
                    'occluded': None,
                    'truncated': None,
                    'bbox_x1': None,
                    'bbox_y1': None,
                    'bbox_x2': None,
                    'bbox_y2': None,
                }
                data_flattened.append({**image_info, **obj_info})
                continue
            for obj in item['labels']:
                obj_info = {
                    'image_id': item['name'],
                    'category': obj['category'],
                    'traffic_light_color': obj['attributes'].get('trafficLightColor', None),
                    'occluded': obj['attributes'].get('occluded', None),
                    'truncated': obj['attributes'].get('truncated', None),
                    'bbox_x1': obj['box2d']['x1'],
                    'bbox_y1': obj['box2d']['y1'],
                    'bbox_x2': obj['box2d']['x2'],
                    'bbox_y2': obj['box2d']['y2']
                }
                # Combine image info and object info
                data_flattened.append({**image_info, **obj_info})
        return pd.DataFrame(data_flattened)

    def _create_dataframe(self, num_samples=-1):
        """
        Creates dataframe for train and validation data
        Args:
            num_samples (int, optional): Defaults to -1.
        """
        if os.path.exists(self.train_csv) and os.path.exists(self.val_csv):
            self.train_df = pd.read_csv(self.train_csv)
            self.val_df = pd.read_csv(self.val_csv)
            self.df_train_val = pd.concat([self.train_df, self.val_df])
        else:
            train_labels = self.load_train_labels()
            val_labels = self.load_val_labels()

            if num_samples > 0:
                train_labels = train_labels[:num_samples]
                val_labels = val_labels[:num_samples]

            self.train_df = self._convert2dataframe(train_labels)
            self.val_df = self._convert2dataframe(val_labels)

            self.train_df['split'] = 'train'
            self.train_df['image_path'] = self.train_images
            # self.train_df = bdd100k_utils.add_image_dimensions(self.train_df,self.train_images)
            self.train_df['img_width'] = 1280
            self.train_df['img_height'] = 720

            self.val_df['split'] = 'val'
            self.val_df['image_path'] = self.val_images
            # self.val_df = bdd100k_utils.add_image_dimensions(self.val_df,self.val_images)
            self.val_df['img_width'] = 1280
            self.val_df['img_height'] = 720

            self.df_train_val = pd.concat([self.train_df, self.val_df])
            self.train_df.to_csv(self.train_csv, index=False)
            self.val_df.to_csv(self.val_csv, index=False)

    def get_dataframes(self):
        if self.train_df is None or self.val_df is None:
            self._create_dataframe()  # Ensure dataframes are created if not already
        return self.train_df, self.val_df

    def plot_distribution(
            self,
            df: pd.DataFrame,
            group_by_column: str,
            title: str,
            output_filename: str,
            x_label: str,
    ):
        """
        Plot the distribution of a given column as percentages in both train and validation datasets.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            group_by_column (str): The column to group by (e.g., 'weather', 'scene').
            title (str): The title of the plot.
            output_filename (str): The name of the file to save the plot.
            x_label (str): The label for the x-axis.
        """
        # Group by the given column and 'split', and count unique 'image_id'
        counts = df.groupby([group_by_column, 'split'])['image_id'].nunique().reset_index(name='count')

        # Calculate percentages for each split
        total_counts = counts.groupby('split')['count'].transform('sum')  # Total count per split
        counts['percentage'] = (counts['count'] / total_counts) * 100  # Convert to percentage

        # Add a column for displaying both percentage and count
        counts['text'] = counts.apply(
            lambda row: f"{row['count']} ({row['percentage']:.2f}%)", axis=1
        )

        # Create a grouped bar plot
        fig = px.bar(
            counts,
            x=group_by_column,
            y='percentage',
            color='split',
            barmode='group',
            title=title,
            labels={
                'percentage': 'Percentage of Images (%)',
                group_by_column: x_label,
                'split': 'Dataset',
            },
            text='text',  # Display both count and percentage on the bars
        )

        # Update layout for better visualization
        fig.update_traces(texttemplate='%{text}', textposition='outside')  # Format text to show count and percentage
        fig.update_layout(
            yaxis=dict(title='Percentage of Images (%)', tickformat=".2f"),
            margin=dict(l=40, r=40, t=40, b=40)
        )

        # Save the plot to the output directory
        png_path = f"{self.output_path}/{output_filename}.png"
        fig.write_image(png_path, format="png", width=800, height=600)

        # Show the plot
        fig.show()

    def plot_weather_distribution(self, df: pd.DataFrame):
        """_summary_

        Args:
            df (pd.DataFrame): _description_
        """
        self.plot_distribution(
            df=df,
            group_by_column='weather',
            title="Percentage Distribution of Weather Conditions",
            output_filename="weather_distribution_percentage",
            x_label="Weather Condition"
        )

    def plot_scene_distribution(self, df: pd.DataFrame):
        """_summary_

        Args:
            df (pd.DataFrame): _description_
        """
        self.plot_distribution(
            df=df,
            group_by_column='scene',
            title="Percentage Distribution of Scene",
            output_filename="scene_distribution_percentage",
            x_label="Scene"
        )

    def plot_time_distribution(self, df: pd.DataFrame):
        """_summary_

        Args:
            df (pd.DataFrame): _description_
        """
        self.plot_distribution(
            df=df,
            group_by_column='timeofday',
            title="Percentage Distribution of Time of day",
            output_filename="timeofday_distribution_percentage",
            x_label='time_of_day'
        )

    def plot_image_without_labels(self, df: pd.DataFrame, max_images=10, images_root_path=None):
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            max_images (int, optional): _description_. Defaults to 10.
            images_root_path (_type_, optional): _description_. Defaults to None.
        """
        images_without_labels = df[df['category'].isna()]['image_id'].unique()
        print(f'{len(images_without_labels)} images do not have any labels/ground truth')
        if len(images_without_labels) > 0:
            max_images = min(len(images_without_labels), max_images)
            images_to_display = images_without_labels[:max_images]
            imgs = []
            for image_name in images_to_display:
                if images_root_path:
                    img_path = file_utils.find_image_and_read(images_root_path, image_name)
                    try:
                        img = decode_image(img_path)
                        imgs.append(img)
                    except:  # noqa: E722
                        pass
            grid = make_grid(imgs)
            plt.figure(figsize=(15, 8))
            plt.axis("off")
            plt.title("Images Without Labels")
            plt.imshow(grid.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C) for matplotlib
            plt.show()

    def plot_column_distribution(self, df: pd.DataFrame, column_name: str, output_path: Optional[str] = None):
        """
        Plot the percentage distribution of a specified column in the dataset using Plotly.
        Args:
            df (pd.DataFrame): The BDD100K dataframe.
            column_name (str): The column to calculate and plot the distribution for.
            output_path (Optional[str]): Path to save the plot as an HTML file. If None, it will only display the plot.
        """
        # Validate that the column exists in the DataFrame
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

        # Calculate the total count for percentage calculation
        # total_count = len(df)

        # Group by the column and calculate counts
        # column_counts = df.groupby(column_name).size().reset_index(name='count')
        column_counts = df.groupby([column_name, 'split']).size().reset_index(name='count')
        total_counts = column_counts.groupby('split')['count'].transform('sum')
        # Calculate percentages
        column_counts['percentage'] = (column_counts['count'] / total_counts) * 100

        # Create the bar chart using Plotly
        fig = px.bar(
            column_counts,
            x=column_name,
            y='percentage',
            color='split',
            barmode='group',
            title=f"Percentage Distribution of '{column_name.capitalize()}'",
            labels={
                'percentage': 'Percentage of Images (%)',
                column_name: column_name.capitalize(),
                'split': 'Dataset'
            },
            text='percentage'  # Show only percentages on the bars
        )

        # Update layout for better visualization
        fig.update_traces(
            texttemplate='%{text:.2f}%',  # Format text as percentage
            textposition='outside'  # Place text outside bars
        )
        fig.update_layout(
            yaxis=dict(title='Percentage of Images (%)', tickformat=".2f"),
            margin=dict(l=40, r=40, t=40, b=40)
        )

        # Show the plot
        fig.show()

        # Save the plot to the output path if specified
        if output_path:
            fig.write_html(f"{output_path}/{column_name}_distribution.html")

    def plot_category_distribution(self, df: pd.DataFrame):
        """
        Plot the distribution of categories as percentages in both train and validation datasets,
        grouped by split without counting unique image IDs.
        """
        # Group by 'category' and 'split', and count occurrences
        category_counts = df.groupby(['category', 'split']).size().reset_index(name='count')

        # Calculate percentages for each split
        total_counts = category_counts.groupby('split')['count'].transform('sum')  # Total count per split
        category_counts['percentage'] = (category_counts['count'] / total_counts) * 100  # Convert to percentage

        # Add a column for displaying both percentage and count
        category_counts['text'] = category_counts.apply(
            lambda row: f"{row['count']} ({row['percentage']:.2f}%)", axis=1
        )

        # Create a grouped bar plot to show the category distribution for both train and val datasets as percentages
        fig = px.bar(
            category_counts,
            x='category',
            y='percentage',
            color='split',
            barmode='group',
            title="Percentage Distribution of Categories",
            labels={
                'percentage': 'Percentage of Images (%)',
                'category': 'Category',
                'split': 'Dataset'
            },
            text='text'  # Display both count and percentage on the bars
        )

        # Update layout for better visualization
        fig.update_traces(texttemplate='%{text}', textposition='outside')  # Format text to show count and percentage
        fig.update_layout(
            yaxis=dict(title='Percentage of Images (%)', tickformat=".2f"),
            margin=dict(l=40, r=40, t=40, b=40)
        )

        # Save the plot to the output directory
        png_path = f"{self.output_path}/category_distribution_percentage.png"
        fig.write_image(png_path, format="png", width=800, height=600)

        # Show the plot
        fig.show()

    def plot_cooccurrence_heatmap(self, df: pd.DataFrame, col1: str, col2: str,
                                  figsize: tuple = (8, 6), cmap: str = 'Blues',
                                  annot_fmt: str = '.2f', title: str = None,
                                  save_path: str = None) -> None:
        """"
        Generates a co-occurrence heatmap for two categorical columns in a DataFrame.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing the data.
        - col1 (str): The name of the first column to analyze.
        - col2 (str): The name of the second column to analyze.
        - figsize (tuple): The size of the plot (width, height). Default is (8, 6).
        - cmap (str): The colormap to use for the heatmap. Default is 'Blues'.
        - annot_fmt (str): The format for the annotations. Default is '.2f' (two decimal places).
        - title (str): The title for the heatmap. Default is None, in which case the title is generated from column names.
        - save_path (str): If provided, the path to save the plot as an image file. Default is None (plot is shown but not saved).

        Returns:
        - None. Displays or saves the heatmap plot.
        """

        # Validate input columns
        if col1 not in df.columns or col2 not in df.columns:
            raise ValueError(f"Columns '{col1}' or '{col2}' not found in DataFrame.")

        # Check if the columns are categorical or need to be converted
        if not pd.api.types.is_categorical_dtype(df[col1]):
            df[col1] = df[col1].astype('category')
        if not pd.api.types.is_categorical_dtype(df[col2]):
            df[col2] = df[col2].astype('category')
        col1_counts = df[col1].value_counts()
        # Compute the co-occurrence counts using crosstab
        co_occurrence = pd.crosstab(df[col1], df[col2])

        # Normalize to get percentages
        total_count = co_occurrence.sum().sum()
        co_occurrence_percent = co_occurrence.div(col1_counts, axis=0) * 100
        # co_occurrence_percent = (co_occurrence / total_count) * 100
        # Prepare annotations and hover data
        annotations = co_occurrence_percent.round(2).astype(str) + '%'
        # hover_data = co_occurrence.astype(str) + ' count<br>' + co_occurrence_percent.round(2).astype(str) + '%'
        hover_data = co_occurrence.applymap(lambda x: f"{x} count<br>{(x / co_occurrence.sum().sum()) * 100:.2f}%")
        # Create the Plotly heatmap
        fig = go.Figure(data=go.Heatmap(
            z=co_occurrence_percent.values,
            x=co_occurrence_percent.columns,
            y=co_occurrence_percent.index,
            colorscale=cmap,  # Color scale
            colorbar=dict(title="Percentage (%)"),  # Colorbar title
            customdata=hover_data.values,
            # hovertemplate='Co-occurrence: %{y} and %{x}<br>Percentage: %{z:.2f}%',  # Hover text
            hovertemplate='Co-occurrence: %{y} and %{x}<br>Count: %{customdata[0]}<br>Percentage: %{customdata[1]}%',
            showscale=True
        ))
        # Add annotations for the percentage with % symbol
        fig.update_traces(
            text=annotations,
            texttemplate='%{text}',  # Show text annotations (percentage) inside cells
            textfont=dict(size=12, color='black'),
            hoverinfo="text"  # Hover will show the percentage and count
        )

        # Update layout to include title and axis labels
        fig.update_layout(
            title=title or f'Co-occurrence Heatmap of {col1} vs {col2}',
            xaxis_title=col2,
            yaxis_title=col1,
            template='plotly_dark',  # Optional: Choose a plotly template (e.g., 'plotly_dark')
        )

        # Show or save the plot
        if save_path:
            fig.write_image(save_path)
            print(f"Heatmap saved to {save_path}")
        else:
            fig.show()

    def create_category_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates a summary DataFrame with the category, the number of samples,
        and the percentage of total samples for each category.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing the data.

        Returns:
        - pd.DataFrame: A new DataFrame with category, number of samples, and percentage.
        """

        # Step 1: Count the number of samples for each category
        category_counts = df['category'].value_counts()

        # Step 2: Calculate the percentage of samples for each category
        total_samples = len(df)
        category_percentage = (category_counts / total_samples) * 100
        category_percentage = category_percentage.round(3)

        # Step 3: Create a new DataFrame with 'category', 'number of samples' and 'percentage'
        summary_df = pd.DataFrame({
            'category': category_counts.index,
            'number_of_samples': category_counts.values,
            'percentage_of_samples': category_percentage.values
        })

        # Optional: Sort by number of samples or percentage (descending)
        summary_df = summary_df.sort_values(by='number_of_samples', ascending=False).reset_index(drop=True)

        return summary_df

    def plot_occlusion_truncation_by_category_percent(self, df: pd.DataFrame):
        """
        Plot the percentage distribution of occlusion and truncation per category for each split
        (train and validation datasets) as stacked bar plots.

        Args:
            df (pd.DataFrame): Input DataFrame containing the data.
        """
        # Validate necessary columns
        required_columns = {'category', 'occluded', 'truncated', 'split'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Iterate over splits (train and validation) and create separate plots
        for split in df['split'].unique():
            # Filter data for the current split
            split_df = df[df['split'] == split]

            # Group by category, occluded, and truncated to calculate counts
            grouped_data = split_df.groupby(['category', 'occluded', 'truncated']).size().reset_index(name='count')

            # Calculate percentages
            total_counts = grouped_data.groupby('category')['count'].transform('sum')  # Total count per category
            grouped_data['percentage'] = (grouped_data['count'] / total_counts) * 100  # Convert to percentage

            # Plot using a stacked bar chart with facets for 'truncated'
            fig = px.bar(
                grouped_data,
                x='category',
                y='percentage',
                color='occluded',
                facet_col='truncated',
                title=f'Percentage of Occlusion and Truncation per Category ({split.capitalize()} Set)',
                labels={
                    'percentage': 'Percentage (%)',
                    'category': 'Category',
                    'occluded': 'Occlusion Status',
                    'truncated': 'Truncation Status'
                },
                text='percentage'  # Show the percentage as text on the bars
            )

            # Update layout for better visualization
            fig.update_traces(
                texttemplate='%{text:.2f}%',
                textposition='inside'
            )
            fig.update_layout(
                yaxis=dict(title='Percentage (%)', tickformat=".2f"),
                margin=dict(l=40, r=40, t=40, b=40)
            )

            # Save the plot to the output directory
            png_path = f"{self.output_path}/occlusion_truncation_category_{split}.png"
            fig.write_image(png_path, format="png", width=1200, height=600)

            # Show the plot
            fig.show()

    def plot_multi_attribute_distribution(
            self,
            df: pd.DataFrame,
            category_column: str = 'category',
            attribute_col_1: str = 'occluded',
            attribute_col_2: str = 'truncated',
            split_column: str = 'split',
            title: str = 'Occlusion and Truncation per Category'
    ):
        """
        Plot the occlusion and truncation distribution by category with percentages.
        Supports visualization by split (train/val).

        Args:
            df (pd.DataFrame): The input DataFrame containing the data.
            category_column (str): The column name for category grouping (e.g., 'category').
            attribute_col_1 (str): The column name for occlusion (e.g., 'occluded').
            attribute_col_2 (str): The column name for truncation (e.g., 'truncated').
            split_column (str): The column name for dataset split (e.g., 'train', 'val').
            title (str): The title of the plot.
        """
        title = f'{attribute_col_1} and {attribute_col_2} per {category_column}'
        # Group by category, occlusion, truncation, and split
        grouped_data = df.groupby([category_column, attribute_col_1, attribute_col_2, split_column]).size().reset_index(
            name='count')

        # Calculate the percentage for each group by category and split
        total_counts = grouped_data.groupby([category_column, split_column])['count'].transform('sum')
        grouped_data['percentage'] = (grouped_data['count'] / total_counts) * 100

        # Create the plot (with or without facet)
        fig = px.bar(
            grouped_data,
            x=category_column,
            y='percentage',
            color=attribute_col_1,
            facet_col=attribute_col_2,  # Separate by truncation column
            barmode='stack',  # Stack bars for occlusion
            title=title,
            labels={
                'percentage': 'Percentage of Objects (%)',
                category_column: category_column.capitalize(),
                attribute_col_1: attribute_col_1,
                attribute_col_2: attribute_col_2,
                split_column: 'Dataset Split'
            },
            text='percentage'  # Display percentage on the bars
        )

        # Update layout for better visualization
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='inside')
        fig.update_layout(
            yaxis=dict(title='Percentage of Objects (%)', tickformat=".2f"),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        output_file = f"{self.output_path}/{attribute_col_1}_{attribute_col_2}_distribution.png"
        fig.write_image(output_file, format="png")

        # Show the plot
        fig.show()


if __name__ == "__main__":
    # Load from a YAML configuration
    config_path = "config.yml"
    bdd_eda = BDD100KEDA(config=config_path)

    # Perform EDA
    print(bdd_eda)
    bdd_eda.explore_label_distribution()
