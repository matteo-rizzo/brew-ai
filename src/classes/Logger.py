import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table


class Logger:
    """
    A custom Logger class using rich for formatted logging.

    Provides logging functionalities with rich formatting for enhanced readability.
    Supports log levels: INFO, DEBUG, WARNING, ERROR, and CRITICAL.
    """

    def __init__(self, name: str = "rich_logger"):
        """
        Initialize the Logger class with rich configuration.

        :param name: Name of the logger instance (default: "rich_logger")
        :type name: str
        """
        # Set up a rich console for custom use
        self.console = Console()

        # Set up a logger with rich handler
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Set logger level to the lowest to capture all logs

        # Set up the rich logging handler
        rich_handler = RichHandler(rich_tracebacks=True, console=self.console)
        formatter = logging.Formatter("%(message)s", datefmt="[%X]")
        rich_handler.setFormatter(formatter)

        # Prevent duplicate logs
        if not self.logger.hasHandlers():
            self.logger.addHandler(rich_handler)

    def info(self, message: str):
        """Log an informational message using rich print."""
        self.console.print(f"[bold green]INFO:[/bold green] {message}")

    def debug(self, message: str):
        """Log a debug message using rich print."""
        self.console.print(f"[bold blue]DEBUG:[/bold blue] {message}")

    def warning(self, message: str):
        """Log a warning message using rich print."""
        self.console.print(f"[bold yellow]WARNING:[/bold yellow] {message}")

    def error(self, message: str):
        """Log an error message using rich print."""
        self.console.print(f"[bold red]ERROR:[/bold red] {message}")

    def critical(self, message: str):
        """Log a critical error message using rich print."""
        self.console.print(f"[bold magenta]CRITICAL:[/bold magenta] {message}")

    def log_panel(self, title: str, content: str):
        """Log a message inside a rich-styled panel for better visualization."""
        panel = Panel(content, title=title, expand=False, border_style="bright_green")
        self.console.print(panel)

    def log_metrics_table(self, model_name: str, metrics: dict):
        """
        Log evaluation metrics in a table format using rich table.

        :param model_name: The name of the model for which metrics are logged.
        :type model_name: str
        :param metrics: A dictionary of metrics where keys are metric names and values are their respective scores.
        :type metrics: dict
        """
        # Create a rich Table object
        table = Table(title=f"Evaluation Metrics for [bold blue]{model_name}[/bold blue]", show_lines=True)

        # Add columns for Metric and Value
        table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="magenta")

        # Add rows to the table for each metric
        for metric_name, metric_value in metrics.items():
            table.add_row(metric_name, f"{metric_value:.4f}")

        # Print the table using the rich console
        self.console.print(table)
