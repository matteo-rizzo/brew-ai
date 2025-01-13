import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError


class DatabaseManager:
    """
    A class to manage database interactions.
    """

    def __init__(self, db_url, verbose=False):
        self.db_url = db_url
        self.verbose = verbose
        self.engine = create_engine(db_url)

    def table_exists(self, table_name):
        """
        Check if a table exists in the database.

        :param table_name: Name of the table to check.
        :return: True if the table exists, False otherwise.
        """
        inspector = inspect(self.engine)
        return table_name in inspector.get_table_names()

    def load_data(self, table_name):
        """
        Load data from a database table into a pandas DataFrame.

        :param table_name: Name of the table to read data from.
        :return: A pandas DataFrame containing the table data.
        """
        try:
            if not self.table_exists(table_name):
                raise ValueError(f"Table '{table_name}' does not exist in the database.")

            if self.verbose:
                print(f"Connecting to database at {self.db_url} and loading table '{table_name}'...")

            with self.engine.connect() as connection:
                data = pd.read_sql_table(table_name, connection)

            if self.verbose:
                print("Data successfully loaded from database.")
            return data
        except SQLAlchemyError as e:
            raise ValueError(f"Database error: {e}") from e
        except Exception as e:
            raise ValueError(f"Unexpected error while loading data from database: {e}") from e

    def load_csv_into_table(self, csv_file, table_name, rename_map=None, drop_unnamed=True, if_exists='replace'):
        """
        Read a CSV file and create/replace a table in the database, letting Pandas auto-infer column types.

        :param csv_file: Path to the CSV file.
        :param table_name: Name of the table to create or replace.
        :param rename_map: Optional dict for renaming columns with special characters, etc.
        :param drop_unnamed: Whether to drop an 'Unnamed: 0' column if present.
        :param if_exists: 'replace' or 'append' or 'fail'. If 'replace', the table is dropped and recreated.
        """
        try:
            if self.verbose:
                print(f"Loading CSV from {csv_file} into table '{table_name}' with if_exists='{if_exists}'...")

            # Read the CSV
            df = pd.read_csv(csv_file)
            if df.empty:
                raise ValueError("CSV appears to be empty.")

            # Optionally drop "Unnamed: 0" if it exists
            if drop_unnamed and 'Unnamed: 0' in df.columns:
                df.drop(columns='Unnamed: 0', inplace=True)

            # Optionally rename columns (for example, removing % or spaces)
            if rename_map:
                df.rename(columns=rename_map, inplace=True)

            # Let Pandas auto-create or replace the table
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)

            if self.verbose:
                print(f"Table '{table_name}' now has {len(df)} rows.")
        except SQLAlchemyError as e:
            raise ValueError(f"Database error: {e}") from e
        except Exception as e:
            raise ValueError(f"Unexpected error loading CSV into table: {e}") from e

    def save_predictions_replace(self, predictions, table_name):
        """
        Always REPLACE any existing table with a new one for the predictions.

        :param predictions: A pandas DataFrame containing the predictions.
        :param table_name: Name of the table to write data to.
        """
        try:
            if self.verbose:
                print(f"Replacing table '{table_name}' in {self.db_url} with new predictions...")

            # to_sql with 'replace' drops & recreates the table
            predictions.to_sql(table_name, self.engine, if_exists='replace', index=False)

            if self.verbose:
                print("Predictions successfully replaced in the database.")
        except SQLAlchemyError as e:
            raise ValueError(f"Database error: {e}") from e
        except Exception as e:
            raise ValueError(f"Unexpected error while saving predictions: {e}") from e

    def save_predictions_append(self, predictions, table_name):
        """
        If the table doesn't exist, create it. Otherwise, append new predictions.

        :param predictions: A pandas DataFrame containing the predictions.
        :param table_name: Name of the table to write data to.
        """
        try:
            if self.verbose:
                print(f"Appending predictions to table '{table_name}' in {self.db_url}...")

            # Create the table if it doesn't exist
            if not self.table_exists(table_name):
                if self.verbose:
                    print(f"Table '{table_name}' does not exist; creating schema from predictions...")
                predictions.head(0).to_sql(table_name, self.engine, index=False)

            predictions.to_sql(table_name, self.engine, if_exists='append', index=False)

            if self.verbose:
                print("Predictions successfully appended to the database.")
        except SQLAlchemyError as e:
            raise ValueError(f"Database error: {e}") from e
        except Exception as e:
            raise ValueError(f"Unexpected error while appending predictions: {e}") from e
