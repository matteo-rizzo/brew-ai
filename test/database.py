import os
import sys
import tempfile
import unittest

import pandas as pd
from sqlalchemy import create_engine, MetaData

sys.path.append(os.path.abspath('..'))
from src.classes.data.DatabaseManager import DatabaseManager

# Optional: rename map for columns in your CSV that have special characters
RENAME_MAP = {
    "Alcol_peso": "Alcol_peso",
    "Alcol_vol": "Alcol_vol",
    "Attenuazione Reale (RDF) %": "Attenuazione_Reale_RDF_pct",
    "Attenuazione vendita apparente %p": "Attenuazione_vendita_apparente_pct",
    "Cellule all'insemenzamento": "Cellule_all_insemenzamento",
    "Diacetile + precursori (ferm.)": "Diacetile_precursori_ferm",
    "Differenza apparente-limite": "Differenza_apparente_limite",
    "Durata di conservazione lievito in cella": "Durata_conservazione_lievito_cella",
    "Estratto apparente": "Estratto_apparente",
    "Estratto apparente limite %Pp": "Estratto_apparente_limite_pct",
    "Estratto reale ": "Estratto_reale",
    "Fermentation rate (13.5Â°-5.5Â°)": "Fermentation_rate_13p5_5p5",
    "Grado primitivo %Pp": "Grado_primitivo_pct",
    "Hopped Wort": "Hopped_Wort",
    "Hopped Wort (37Â°C)": "Hopped_Wort_37C",
    "NBB-A": "NBB_A",
    "NBB-B": "NBB_B",
    "Pentandione + precursori": "Pentandione_precursori",
    "pH": "pH",
    "Rapporto ER/ABW": "Rapporto_ER_ABW",
    "Temperatura Fermentazione": "Temperatura_Fermentazione",
    "Temp. lievito all'insemenzamento": "Temp_lievito_insemenzamento",
    "Tempo di riduzione diacetile": "Tempo_riduzione_diacetile"
}


class TestDatabaseFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False).name
        cls.db_url = f"sqlite:///{cls.db_file}"
        cls.engine = create_engine(cls.db_url)
        cls.metadata = MetaData()
        cls.db_manager = DatabaseManager(cls.db_url, verbose=True)

        cls.csv_file = '../dataset/beer-fermentation.csv'
        cls.predictions_file = '../logs/predictions.csv'

    def setUp(self):
        """Drop and clear everything before each test; do NOT load CSV by default."""
        self.table_name = f"mock_table_{self._testMethodName}"
        self.metadata.drop_all(self.engine)
        self.metadata.clear()

    def load_csv_into_table(self):
        """
        Helper method to let Pandas create/replace the table from the CSV.
        """
        df = pd.read_csv(self.csv_file)
        if 'Unnamed: 0' in df.columns:
            df.drop(columns='Unnamed: 0', inplace=True)
        df.rename(columns=RENAME_MAP, inplace=True)
        df.to_sql(self.table_name, self.engine, if_exists='replace', index=False)

    def test_01_load_data_from_database(self):
        """Now we manually load the CSV so the table exists and has data."""
        self.load_csv_into_table()

        data = self.db_manager.load_data(self.table_name)
        self.assertGreater(len(data), 0, "No data loaded from the database.")
        self.assertIn("Alcol_peso", data.columns, "Alcol_peso column is missing.")

    def test_02_empty_table_handling(self):
        """
        Here we do NOT call load_csv_into_table(), so no table is created.
        Attempting to load it should raise ValueError.
        """
        with self.assertRaises(ValueError, msg="Expected error if table doesn't exist"):
            self.db_manager.load_data(self.table_name)

    def test_03_save_predictions_to_database(self):
        """Load CSV, then overwrite with predictions."""
        self.load_csv_into_table()

        preds = pd.read_csv(self.predictions_file, header=None)
        preds.columns = ['Prediction']
        preds.insert(0, 'id', [str(i + 1) for i in range(len(preds))])

        # This uses a REPLACE or APPEND logic depending on your DatabaseManager method
        self.db_manager.save_predictions_replace(preds, self.table_name)

        data = self.db_manager.load_data(self.table_name)
        self.assertEqual(len(data), len(preds), "Mismatch after saving predictions.")

    @classmethod
    def tearDownClass(cls):
        """Remove DB file."""
        cls.metadata.drop_all(cls.engine)
        if os.path.exists(cls.db_file):
            os.remove(cls.db_file)


if __name__ == '__main__':
    unittest.main()
