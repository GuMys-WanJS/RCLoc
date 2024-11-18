import os
import pandas as pd


class IverilogDataLoader:
    def __init__(self, bug_id, feature_df, label_df):
        self.bug_id = bug_id
        self.feature_df = feature_df
        self.label_df = label_df
        self.data_df = None
        self.fault_file = None
        self.file_dir = None
        self.rest_columns = []

    def load(self):
        self._load_features()

    def _load_features(self):
        self.data_df = pd.concat([self.feature_df, self.label_df], axis=1)




