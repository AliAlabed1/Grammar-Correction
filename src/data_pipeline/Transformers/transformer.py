import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from src.utils.logging_utils import app_logger


class GrammarCorrectionDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        ungrammatical_statement = self.dataframe.iloc[idx]["Ungrammatical Statement"]
        standard_english = self.dataframe.iloc[idx]["Standard English"]
        return ungrammatical_statement, standard_english
    
class Transformer():
    def transform(self,df:pd.DataFrame):
        """
        Transforms the input DataFrame and returns a new DataFrame.

        Args:
            - df (pd.DataFrame): The input DataFrame to transform.

        Returns:
            - pd.DataFrame: The transformed DataFrame.

        Rises:
            - TypeError: If the df is not a dataframe.
            - ValueError: If the df df is None.
        """
        if df is None:
            app_logger.error("The df is None")
            raise ValueError("The loaded df is None.")
        
        
        df_clean = df.drop_duplicates()
        df_clean['Ungrammatical Statement'] = df_clean['Ungrammatical Statement'].str.strip()
        df_clean['Standard English'] = df_clean['Standard English'].str.strip()

        train_df, test_df = train_test_split(df_clean, test_size = 0.2)
        train_df, val_df = train_test_split(train_df, test_size = 0.1)


        train_dataset = GrammarCorrectionDataset(train_df)
        val_dataset = GrammarCorrectionDataset(val_df)
        test_dataset = GrammarCorrectionDataset(test_df)

        
        
        
        app_logger.info("DF is transformed successfully.")
        return train_dataset,val_dataset,test_dataset


# Test the Class and main Funcation 
if __name__ == "__main__":
    itransform = Transformer()
    df = itransform.transform(pd.DataFrame())

