# -*- coding: utf-8 -*-
import os
from datetime import datetime

# PATH
csv_path = 'DATABASE TESI CHIARA TIBALDI.xlsx - Database.csv'
output_root = './tesi_tebaldi_results'
test_folder = os.path.join(output_root, f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
os.makedirs(test_folder, exist_ok=True)

# COLONNE
target_col = 'final_diagnosis (0-4)'
t_1_visit = 'diagn_1_vis'
t_2_visit = 'second_opinion (0/1)'
id_cols = ['ID', 'Nome', 'Cognome', 'CODALS', 'gene_mut',
           # date
           'birth_date_month', 'birth_date_day', 'birth_date_year',
           'visit_1 (data)_year', 'visit_1 (data)_month', 'visit_1 (data)_day',
           'date_onset_year', 'date_onset_month', 'date_onset_day',
           'date_diagnosis_year', 'date_diagnosis_month', 'date_diagnosis_day',
           'MRC_T0_date_year', 'MRC_T0_date_month', 'MRC_T0_date_day',
           'birth_date', 'visit_1 (data)', 'date_onset', 'date_diagnosis', 'MRC_T0_date',
           # altre colonne
            'second_opinion (0/1)', 'diagn_1_vis',
            'MRC_T0_neck_flex', 'MRC_T0_neck_est', 'MRC_T0_delt_R', 'MRC_T0_wr_flex_R',
            'MRC_T0_wr_flex_L', 'MRC_T0_hip_flex_R', 'MRC_T0_ankle_ext_R', 'MRC_T0_ankle_ext_L',
            'MRC_T0_delt_L', 'MRC_T0_BB_R', 'MRC_T0_wr_est_R', 'MRC_T0_wr_est_L', 'MRC_T0_hip_flex_L',
           'MRC_T0_leg_ext_R', 'MRC_T0_ankle_flex_R', 'MRC_T0_ankle_flex_L',
            'MRC_T0_BB_L', 'MRC_T0_TB_R', 'MRC_T0_fing_flex_R', 'MRC_T0_fing_flex_L',
            'MRC_T0_fing_est_R', 'MRC_T0_fing_est_L', 'MRC_T0_leg_ext_L', 'MRC_T0_leg_flex_R',
            'MRC_T0_TB_L', 'MRC_T0_thumb_R', 'MRC_T0_thumb_L', 'MRC_T0_leg_flex_L',

            'R_prox_UL_atrophy', 'L_prox_UL_atrophy', 'R_dist_UL_atrophy', 'L_dist_UL_atrophy', 'R_prox_LL_atrophy',
           'L_prox_LL_atrophy', 'R_dist_LL_atrophy', 'L_dist_LL_atrophy', 'R_brisk_biceps', 'R_brisk_patellar',
           'R_brisk_ankle', 'R_UL_MAS_Penn', 'R_LL_MAS_Penn', 'R_brisk_biceps_calc', 'R_brisk_brachioradialis_calc',
           'R_brisk_patellar_calc', 'R_brisk_ankle_calc_', 'jaw_jerk', 'R_Hoffman', 'L_Hoffman', 'R_Babinski',
           'L_Babinski', 'R_palmomental', 'R_UL_spasticity_as_Penn', 'L_UL_spasticity_as_Penn',
           'R_LL_spasticity_as_Penn', 'L_LL_spasticity_as_Penn',
            'L_brisk_biceps', 'L_brisk_patellar', 'L_brisk_ankle',
           'L_UL_MAS_Penn', 'L_LL_MAS_Penn', 'L_brisk_biceps_calc',
           'L_brisk_patellar_calc', 'L_brisk_ankle_calc', 'L_palmomental',

'alsfrs_bulb', 'alsfrs_aass', 'alsfrs_aaii', 'alsfrs_resp.',  'em_lability (0/1)',
'cramps (0/1)', 'fasciculation (0/1)', 'progression (0/1)', 'fvc (%)', 'emg (0/1)',
'eng (0/1)',  'n_site (0/4)', ' ck (valore)', 'IgM Borrelia (0/1)', 'IgG Borrelia (0/1)',
'WB_Borrelia (0/1)', 'brain_mri_mnd (0/1)', 'brain_mri_other (0/1)', 'spine_mri (0/1)',
'pet (0/1)', 'test_neuro (0/1)', 'riluzole_1_vis (0/1)', 'genetic_status (0/1)',
'peg (0/1)', 'niv (0/1)', 'tracheostomy (0/1)', 'tongue_atrophy', 'R_brisk_brachioradialis',
'L_brisk_brachioradialis_calc', 'Turin_tot', 'Turin_lower_MN',

'phenotype (1-8)', 'res (0/1)', 'familiarity_MND/demenza/psich./Paget (0/1)','age_onset (y)',
# 'delay (m)', 'HBW (kg)', '\u0394weight (kg)', 'alsfrs_evalutation_1_visit', '\u0394alsfrs',
# 'alsfrs_bulb', 'alsfrs_aass', 'alsfrs_aaii', 'alsfrs_resp.',  'em_lability (0/1)',
# 'cramps (0/1)', 'fasciculation (0/1)', 'progression (0/1)', 'fvc (%)',
'emg (0/1)', 'eng (0/1)', 'n_site (0/4)', ' ck (valore)', 'IgM Borrelia (0/1)',
'IgG Borrelia (0/1)', 'WB_Borrelia (0/1)', 'brain_mri_mnd (0/1)', 'brain_mri_other (0/1)',
'spine_mri (0/1)', 'pet (0/1)', 'test_neuro (0/1)', 'riluzole_1_vis (0/1)',
'genetic_status (0/1)', 'peg (0/1)', 'niv (0/1)', 'tracheostomy (0/1)', 'tongue_atrophy',
'R_brisk_brachioradialis', 'L_brisk_brachioradialis_calc', 'Turin_tot', 'Turin_lower_MN',

'type_onset (0-4)', 'site_of_onset', 'elescorial_class (0-3)', 'phenotype (1-8)',
           ]

forced_categorical_columns = ['site_of_onset', 'phenotype (1-8)', 'type_onset (0-4)']
forced_numerical_cols = ['res (0/1)', 'familiarity_MND/demenza/psich./Paget (0/1)',
                         'em_lability (0/1)', 'cramps (0/1)',
                         'fasciculation (0/1)', 'progression (0/1)', 'emg (0/1)', 'eng (0/1)',
                         'n_site (0/4)', 'IgM Borrelia (0/1)', 'IgG Borrelia (0/1)', 'WB_Borrelia (0/1)',
                         'brain_mri_mnd (0/1)', 'brain_mri_other (0/1)', 'spine_mri (0/1)', 'pet (0/1)',
                         'test_neuro (0/1)', 'riluzole_1_vis (0/1)', 'genetic_status (0/1)', 'peg (0/1)',
                         'niv (0/1)', 'tracheostomy (0/1)', 'tongue_atrophy',
                         'R_prox_UL_atrophy', 'L_prox_UL_atrophy', 'R_dist_UL_atrophy', 'L_dist_UL_atrophy', 'R_prox_LL_atrophy',
                         'L_prox_LL_atrophy', 'R_dist_LL_atrophy', 'L_dist_LL_atrophy', 'R_brisk_biceps',
                         'R_brisk_brachioradialis', 'R_brisk_patellar', 'L_brisk_patellar', 'R_brisk_ankle',
                         'L_brisk_ankle', 'R_UL_MAS_Penn', 'L_UL_MAS_Penn', 'R_LL_MAS_Penn', 'L_LL_MAS_Penn',
                         'R_brisk_biceps_calc', 'L_brisk_biceps_calc', 'R_brisk_brachioradialis_calc',
                         'L_brisk_brachioradialis_calc', 'R_brisk_patellar_calc', 'L_brisk_patellar_calc',
                         'R_brisk_ankle_calc_', 'L_brisk_ankle_calc', 'jaw_jerk', 'R_Hoffman', 'L_Hoffman',
                         'R_Babinski', 'L_Babinski', 'R_palmomental', 'L_palmomental', 'R_UL_spasticity_as_Penn',
                         'L_UL_spasticity_as_Penn', 'R_LL_spasticity_as_Penn', 'L_LL_spasticity_as_Penn',


                         ]
binary_cols = []  # colonne 0/1 da trattare come numeriche

# PARAMETRI
n_splits = 5
random_state = 42
top_n_features = 2
min_common_models = 5  # per feature comuni
