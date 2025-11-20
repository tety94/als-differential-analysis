# -*- coding: utf-8 -*-
import os
from datetime import datetime

model_output_folder = 'models'
# PATH
csv_path = 'database.csv'
output_root = './results'
test_folder = os.path.join(output_root, f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
os.makedirs(test_folder, exist_ok=True)

# COLONNE
target_col = ''
t_1_visit = ''
id_cols = []

categorical_columns = []
forced_numerical_cols = []
binary_cols = []

# PARAMETRI
n_splits = 5
random_state = 42
top_n_features = 2
min_common_models = 5

#connessione database
DB_USER = ""
DB_PASS = ""
DB_HOST = ""
DB_PORT = "3306"
DB_NAME = ""

SECRET_KEY=""