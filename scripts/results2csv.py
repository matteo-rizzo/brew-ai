import os
import json
import pandas as pd
from rich.console import Console
from rich.progress import track
from rich.table import Table
from rich import box

# Initialize rich console for better printing
console = Console()

# Directory where evaluation results are stored
BASE_DIR = 'results/'
AGGREGATED_CSV_PATH = 'logs/aggregated_results.csv'

def find_metrics_files(base_dir: str) -> pd.DataFrame:
    """
    Traverse the directory and collect all the _metrics.json files.

    :param base_dir: Base directory to search for _metrics.json files
    :return: A DataFrame containing aggregated metrics for each model
    """
    aggregated_data = []

    console.print("[bold cyan]Searching for _metrics.json files in evaluation results...[/bold cyan]")

    # Traverse directories and collect metrics files
    for root, dirs, files in track(os.walk(base_dir), description="Processing directories..."):
        for file in files:
            if file.endswith('_metrics.json'):
                file_path = os.path.join(root, file)
                model_name = os.path.basename(root)

                metrics = load_metrics_from_json(file_path)
                if metrics:
                    metrics['model_name'] = model_name
                    aggregated_data.append(metrics)

    if aggregated_data:
        console.print(f"[bold green]Successfully aggregated metrics from {len(aggregated_data)} models.[/bold green]")
    else:
        console.print("[bold red]No metrics found in the specified directories.[/bold red]")

    return pd.DataFrame(aggregated_data)

def load_metrics_from_json(json_file: str) -> dict:
    """
    Load the contents of a _metrics.json file.

    :param json_file: Path to the _metrics.json file
    :return: Dictionary containing the metrics
    """
    try:
        with open(json_file, 'r') as f:
            metrics = json.load(f)
        console.print(f"[bold yellow]Loaded metrics from:[/bold yellow] {json_file}")
        return metrics
    except (FileNotFoundError, json.JSONDecodeError) as e:
        console.print(f"[bold red]Error loading {json_file}: {e}[/bold red]")
        return {}

def save_to_csv(dataframe: pd.DataFrame, output_path: str) -> None:
    """
    Save the aggregated DataFrame to a CSV file.

    :param dataframe: DataFrame containing the aggregated results
    :param output_path: Path to save the CSV file
    """
    dataframe.to_csv(output_path, index=False)
    console.print(f"[bold green]Aggregated results saved to:[/bold green] {output_path}")

def display_summary_table(dataframe: pd.DataFrame) -> None:
    """
    Display a summary table of the aggregated results using rich.

    :param dataframe: DataFrame containing the aggregated results
    """
    table = Table(title="Aggregated Model Metrics", box=box.ROUNDED)

    # Add column headers
    for col in dataframe.columns:
        table.add_column(col.capitalize(), justify="center")

    # Add rows of data
    for _, row in dataframe.iterrows():
        table.add_row(*[str(val) for val in row.values])

    console.print(table)

def main():
    """
    Main function to aggregate results and save as CSV.
    """
    console.print("[bold cyan]Starting the aggregation process...[/bold cyan]")

    # Step 1: Find all _metrics.json files and aggregate results
    aggregated_df = find_metrics_files(BASE_DIR)

    if not aggregated_df.empty:
        # Step 2: Display the aggregated results in a summary table
        display_summary_table(aggregated_df)

        # Step 3: Save the aggregated results to a CSV file
        save_to_csv(aggregated_df, AGGREGATED_CSV_PATH)
    else:
        console.print("[bold red]No metrics found to aggregate.[/bold red]")

if __name__ == "__main__":
    main()
