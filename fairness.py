import sys
import pandas as pd
from scipy.stats import ttest_ind
from scipy.stats import wasserstein_distance, kstest
import numpy as np
import argparse
import glob
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Example Python script with argument parser"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input file"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output file"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="Path to output file"
    )
    return parser.parse_args()

def _to_numpy(x):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")
    return x

def _rate_exceed(scores, mask, z):
    """P(score >= z | mask)"""
    s = scores[mask]
    if s.size == 0:
        return np.nan
    return np.mean(s >= z)

def save_to_file(content, filename):
    content = content.ravel()
    with open(filename, "w") as opened_file:
        for item in content:
            opened_file.write(str(item))
            opened_file.write("\n")

def read_ethnicity_labels(path):
    subject_dict = {}
    with open(path, "r") as opened_file:
        content = opened_file.readlines()
    for ith_line in content:
        ith_line = ith_line.strip()
        subject_name = ith_line.split(":")[0]
        details = ith_line.split(":")[1]
        info = details.split(", ")
        if info[0].lstrip()=="male":
            gender = 0
        elif info[0].lstrip()=="female":
            gender = 1
        if info[1]=="afro_american":
            ethnicity = 0
        elif info[1]=="asian":
            ethnicity = 1
        elif info[1]=="indian":
            ethnicity = 1
        elif info[1]=="caucasian":
            ethnicity = 2
        elif info[1]=="other":
            ethnicity = -1
        else:
            continue
        subject_dict[subject_name] = [ethnicity, gender]
    return subject_dict

def remove_outliers_iqr_per_type(df, value_col, type_col='label'):
    def iqr_filter(group):
        Q1 = group[value_col].quantile(0.25)
        Q3 = group[value_col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return group[(group[value_col] >= lower) & (group[value_col] <= upper)]

    return df.groupby(type_col, group_keys=False).apply(iqr_filter)

def remove_outliers(df):
    Q1 = df['mean_error'].quantile(0.25)
    Q3 = df['mean_error'].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df_clean = df[(df['mean_error'] >= lower) & (df['mean_error'] <= upper)]
    return df_clean

def visualize_dataframe(df):
    df.to_excel(args.output, index=False)

def calculate_cv(x1, x2):
    cv = np.cov(x1, x2, ddof=1)
    return cv

def normalized_wasserstein(x, y):
    all_vals = np.concatenate([x, y])
    min_v = all_vals.min()
    max_v = all_vals.max()
    if max_v == min_v:
        return 0.0
    x_norm = (x - min_v) / (max_v - min_v)
    y_norm = (y - min_v) / (max_v - min_v)
    return wasserstein_distance(x_norm, y_norm)

if __name__ == "__main__":
    args = parse_args()
    if args.mode=="ethnicity":
        select_pairs = [{'asian': 'caucasian'}, {'afro_american': 'asian'}, {'caucasian': 'afro_american'}]
        subject_maps = {"afro_american": 0, "asian": 1, "caucasian": 2}
    elif args.mode=="gender":
        select_pairs = [{'male': 'female'}]
        subject_maps = {"male": 0, "female": 1}
    
    t_values = []
    p_values = []
    sp_values = []
    di_values = []
    cv_values = []
    e1_values = []
    e2_values = []

    pairs = []
    methods = []
    wasserstein_values = []
    kolmogorov_values = []
    mmds = []
    ks_values = []

    result_root_path = args.input
    result_folders = os.listdir(result_root_path)

    for result_folder in result_folders:
        result_paths = glob.glob(os.path.join(result_root_path, result_folder)+"/**.csv", recursive=True)    
        dataframes = []
        for result_path in result_paths:
            dataframe = pd.read_csv(result_path)
            dataframes.append(dataframe)
        if len(dataframes) == 0:
            continue
        for select_pair in select_pairs:
            selected_variable = list(select_pair.keys())[0]
            selected_other_variable = list(select_pair.values())[0]

            avg_result = pd.concat([df['Error'] for df in dataframes], axis=1).mean(axis=1)
            df_avg = dataframes[0].copy()
            df_avg['Error'] = avg_result

            df_avg = df_avg.rename(columns={'Id': 'subjectname', 'Error': 'mean_error', 'Ethnicity': 'label'})

            dataframe = df_avg

            ethnicities = ["afro_american", "asian", "caucasian"]
            gender = ["male", "female"]
            
            label_type = ["ethnicity", "gender"]

            variables = [ethnicities, gender]

            dataframe = dataframe.replace("black", "afro_american")
            dataframe = dataframe.replace("man", "male")
            dataframe = dataframe.replace("woman", "female")

            X, Y = [], []
            for ith, row in dataframe.iterrows():
                try:
                    subjectname = str(int(row["subjectname"]))
                except:
                    subjectname = str(row["subjectname"])

                label = dataframe.loc[ith, "DemographicLabel"]

                if label=="other":
                    continue

                X.append(row["mean_error"])
                label_mapped = subject_maps[label]
                Y.append(label_mapped)

            X = np.array(X)
            Y = np.array(Y)

            X = X.reshape(-1, 1)

            mi = 0
            group_a = Y==subject_maps[selected_variable]
            group_b = Y==subject_maps[selected_other_variable]

            group_a_members = X[group_a, 0]
            group_b_members = X[group_b, 0]

            wasserstein_value = wasserstein_distance(X[Y==subject_maps[selected_variable], 0], X[Y==subject_maps[selected_other_variable], 0])
            wasserstein_values.append(wasserstein_value)

            ks_value = kstest(X[Y==subject_maps[selected_variable]], X[Y==subject_maps[selected_other_variable]])
            ks_values.append(round(ks_value.statistic[0], 3))
            t_stat, p_val = ttest_ind(X[Y==subject_maps[selected_variable]], X[Y==subject_maps[selected_other_variable]])

            kolmogorov_value = kstest(X[Y==subject_maps[selected_variable]], X[Y==subject_maps[selected_other_variable]]).pvalue[0]
            kolmogorov_values.append(kolmogorov_value)

            t_values.append(t_stat.item())
            p_values.append(p_val.item())
            
            pairs.append(select_pair)
            methods.append(result_folder)

            e1_value = np.mean(X[Y==subject_maps[selected_variable]])
            e2_value = np.mean(X[Y==subject_maps[selected_other_variable]])

            e1_values.append(e1_value)
            e2_values.append(e2_value)

    df = pd.DataFrame({
        "Method": methods,
        "Pairs": pairs,
        "T_value": t_values,
        "P_value": p_values,
        "Wasserstein": wasserstein_values,
        "KS": ks_values,
        "E1": e1_values,
        "E2": e2_values,
    })
    visualize_dataframe(df)
