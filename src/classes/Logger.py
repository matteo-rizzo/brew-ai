import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel


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
        """
        Log an informational message using rich print.

        :param message: The informational message to log
        :type message: str
        """
        self.console.print(f"[bold green]INFO:[/bold green] {message}")

    def debug(self, message: str):
        """
        Log a debug message using rich print.

        :param message: The debug message to log
        :type message: str
        """
        self.console.print(f"[bold blue]DEBUG:[/bold blue] {message}")

    def warning(self, message: str):
        """
        Log a warning message using rich print.

        :param message: The warning message to log
        :type message: str
        """
        self.console.print(f"[bold yellow]WARNING:[/bold yellow] {message}")

    def error(self, message: str):
        """
        Log an error message using rich print.

        :param message: The error message to log
        :type message: str
        """
        self.console.print(f"[bold red]ERROR:[/bold red] {message}")

    def critical(self, message: str):
        """
        Log a critical error message using rich print.

        :param message: The critical error message to log
        :type message: str
        """
        self.console.print(f"[bold magenta]CRITICAL:[/bold magenta] {message}")

    def log_panel(self, title: str, content: str):
        """
        Log a message inside a rich-styled panel for better visualization.

        :param title: The title of the panel
        :type title: str
        :param content: The content to display inside the panel
        :type content: str
        """
        panel = Panel(content, title=title, expand=False, border_style="bright_green")
        self.console.print(panel)
