from sklearn.base import BaseEstimator, TransformerMixin
from typing import Literal

# Class modus imputer global
class ModusImputer(BaseEstimator, TransformerMixin):
    def __init__(self, fill_columns):
        self.fill_columns = fill_columns
        self.fill_values = {}

    def fit(self, X, y=None):
        # iterasi tiap kolom dan ambil modusnya
        self.fill_values = {col: X[col].mode()[0] for col in self.fill_columns}
        return self

    def transform(self, X, y=None):
        # mengisi miss val dengan dict modus
        data = X.copy()
        for col in self.fill_columns:
            data[col] = data[col].fillna(self.fill_values[col])
        return data

    def set_output(self, transform: Literal['default', 'pandas'] = 'default'):
        return super().set_output(transform=transform)
    

# Class modus imputer 2 kolom
class ModusTwoGroups(BaseEstimator, TransformerMixin):
    def __init__(self, fill_columns):
        self.fill_columns = fill_columns
        self.fill_values = {}

    def fit(self, X, y=None):
        # ambil modus kombinasi 2 kolom
        self.fill_values = ( X.groupby(self.fill_columns)
                            .size()
                            .reset_index(name='count')      # beri nama kolom frekuensi
                            .sort_values(by='count', ascending=False)  # urutkan berdasarkan frekuensi tertinggi
                            .iloc[0][self.fill_columns]   # ambil kombinasi dengan frekuensi tertinggi
                        ).to_dict()

        return self

    def transform(self, X, y=None):
        # mengisi miss val dengan dict fill_values
        data = X.copy()
        for col in self.fill_columns:
            data[col] = data[col].fillna(self.fill_values[col])
        return data

    def set_output(self, transform: Literal['default', 'pandas'] = 'default'):
        return super().set_output(transform=transform)