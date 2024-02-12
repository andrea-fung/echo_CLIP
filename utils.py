import torch
import numpy as np
from pathlib import Path
import cv2
import re

import pandas as pd
import matplotlib.pyplot as plt
import imageio as iio
import os

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_array
from typing import List, Union, Tuple

zero_shot_prompts = {
    "ejection_fraction": [
        "THE LEFT VENTRICULAR EJECTION FRACTION IS ESTIMATED TO BE <#>% ",
        "LV EJECTION FRACTION IS <#>%. ",
    ],
    "pacemaker": [
        "ECHO DENSITY IN RIGHT VENTRICLE SUGGESTIVE OF CATHETER, PACER LEAD, OR ICD LEAD. ",
        "ECHO DENSITY IN RIGHT ATRIUM SUGGESTIVE OF CATHETER, PACER LEAD, OR ICD LEAD. ",
    ],
    "impella": [
        "AN IMPELLA CATHETER IS SEEN AND THE INLET AREA IS 4.0CM FROM THE AORTIC VALVE AND DOES NOT INTERFERE WITH NEIGHBORING STRUCTURES, CONSISTENT WITH CORRECT IMPELLA POSITIONING. THERE IS DENSE TURBULENT COLOR FLOW ABOVE THE AORTIC VALVE, CONSISTENT WITH CORRECT OUTFLOW AREA POSITION ",
        "AN IMPELLA CATHETER IS SEEN ACROSS THE AORTIC VALVE AND IS TOO CLOSE TO OR ENTANGLED IN THE PAPILLARY MUSCLE AND SUBANNULAR STRUCTURES SURROUNDING THE MITRAL VALVE; REPOSITIONING RECOMMENDED. ",
        "AN IMPELLA CATHETER IS SEEN, HOWEVER THE INLET AREA APPEARS TO BE IN THE AORTA OR NEAR THE AORTIC VALVE; REPOSITIONING IS RECOMMENDED. ",
        "AN IMPELLA CATHETER IS SEEN ACROSS THE AORTIC VALVE AND EXTENDS TOO FAR INTO THE LEFT VENTRICLE; REPOSITIONING RECOMMENDED ",
    ],
    "normal_right_atrial_pressure": [
        "THE INFERIOR VENA CAVA SHOWS A NORMAL RESPIRATORY COLLAPSE CONSISTENT WITH NORMAL RIGHT ATRIAL PRESSURE (3MMHG). ",
    ],
    "elevated_right_atrial_pressure": [
        "THE INFERIOR VENA CAVA DEMONSTRATES LESS THAN 50% COLLAPSE CONSISTENT WITH ELEVATED RIGHT ATRIAL PRESSURE (8MMHG). ",
    ],
    "significantly_elevated_right_atrial_pressure": [
        "THE INFERIOR VENA CAVA DEMONSTRATES NO INSPIRATORY COLLAPSE, CONSISTENT WITH SIGNIFICANTLY ELEVATED RIGHT ATRIAL PRESSURE (>15MMHG). ",
    ],
    "pulmonary_artery_pressure": [
        "ESTIMATED PA SYSTOLIC PRESSURE IS <#>MMHG. ",
        "ESTIMATED PA PRESSURE IS <#>MMHG. ",
        "PA PEAK PRESSURE IS <#>MMHG. ",
    ],
    "severe_left_ventricle_dilation": [
        "SEVERE DILATED LEFT VENTRICLE BY LINEAR CAVITY DIMENSION. ",
        "SEVERE DILATED LEFT VENTRICLE BY VOLUME. ",
        "SEVERE DILATED LEFT VENTRICLE. ",
    ],
    "moderate_left_ventricle_dilation": [
        "MODERATE DILATED LEFT VENTRICLE BY LINEAR CAVITY DIMENSION. ",
        "MODERATE DILATED LEFT VENTRICLE BY VOLUME. ",
        "MODERATE DILATED LEFT VENTRICLE. ",
    ],
    "mild_left_ventricle_dilation": [
        "MILD DILATED LEFT VENTRICLE BY LINEAR CAVITY DIMENSION. ",
        "MILD DILATED LEFT VENTRICLE BY VOLUME. ",
        "MILD DILATED LEFT VENTRICLE. ",
    ],
    "severe_right_ventricle_size": ["SEVERE DILATED RIGHT VENTRICLE. "],
    "moderate_right_ventricle_size": ["MODERATE DILATED RIGHT VENTRICLE. "],
    "mild_right_ventricle_size": ["MILD DILATED RIGHT VENTRICLE. "],
    "severe_left_atrium_size": ["SEVERE DILATED LEFT ATRIUM. "],
    "moderate_left_atrium_size": ["MODERATE DILATED LEFT ATRIUM. "],
    "mild_left_atrium_size": ["MILD DILATED LEFT ATRIUM. "],
    "severe_right_atrium_size": ["SEVERE DILATED RIGHT ATRIUM. "],
    "moderate_right_atrium_size": ["MODERATE DILATED RIGHT ATRIUM. "],
    "mild_right_atrium_size": ["MILD DILATED RIGHT ATRIUM. "],
    "tavr": [
        "A BIOPROSTHETIC STENT-VALVE IS PRESENT IN THE AORTIC POSITION. ",
    ],
    "mitraclip": [
        "TWO MITRACLIPS ARE SEEN ON THE ANTERIOR AND POSTERIOR LEAFLETS OF THE MITRAL VALVE. ",
        "TWO MITRACLIPS ARE NOW PRESENT ON THE ANTERIOR AND POSTERIOR MITRAL VALVE LEAFLETS. ",
        "ONE MITRACLIP IS SEEN ON THE ANTERIOR AND POSTERIOR LEAFLETS OF THE MITRAL VALVE. ",
    ],
}


def compute_binary_metric(
    video_embeddings: torch.Tensor,
    prompt_embeddings: torch.Tensor,
):
    per_frame_similarities = video_embeddings @ prompt_embeddings.T
    # Average along the candidate dimension and frame dimension
    predictions = per_frame_similarities.mean(dim=-1).mean(dim=-1)

    return predictions


def compute_regression_metric(
    video_embeddings: torch.Tensor,
    prompt_embeddings: torch.Tensor,
    prompt_values: torch.Tensor,
):
    per_frame_similarities = (
        video_embeddings @ prompt_embeddings.T
    )  # (N x Frames x Candidates)

    # Sort the candidates by their similarity to the video
    ranked_candidate_phrase_indices = torch.argsort(
        per_frame_similarities, dim=-1, descending=True
    )

    # Convert matrix of indices to their corresponding continuous values.
    prompt_values = torch.tensor(
        prompt_values, device=video_embeddings.device
    )  # (N x Frames x Candidates)
    all_frames_ranked_values = prompt_values[ranked_candidate_phrase_indices]

    # Taking the mean along dim=1 collapses the frames dimension
    avg_frame_ranked_values = all_frames_ranked_values.float().mean(
        dim=1
    )  # (N x Candidates)

    # The median of only the top 20% of predicted values is taken
    # as the final predicted value
    twenty_percent = int(avg_frame_ranked_values.shape[1] * 0.2)
    final_prediction = avg_frame_ranked_values[:, :twenty_percent].median(dim=-1)[0]

    return final_prediction


def crop_and_scale(img, res=(640, 480), interpolation=cv2.INTER_CUBIC, zoom=0.1):
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding]
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding]
    if zoom != 0:
        pad_x = round(int(img.shape[1] * zoom))
        pad_y = round(int(img.shape[0] * zoom))
        img = img[pad_y:-pad_y, pad_x:-pad_x]

    img = cv2.resize(img, res, interpolation=interpolation)

    return img


def read_avi(p: Path, res=None):
    cap = cv2.VideoCapture(str(p))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if res is not None:
            frame = crop_and_scale(frame, res)
        frames.append(frame)
    cap.release()
    return np.array(frames)


## TEXT CLEANING UTILS

removables = re.compile(r"\^|CRLF|‡")

in_text_periods = re.compile(r"(?<=\D)\.|\.(?=\D)")
square_brackets = re.compile(r"[\[\]]")
multi_whitespace = re.compile(r"\s+")
multi_period = re.compile(r"\.+")

select_was = re.compile(r"(?<=\b)WAS(?=\b)")
select_were = re.compile(r"(?<=\b)WERE(?=\b)")
select_and_or = re.compile(r"(?<=\b)AND/OR(?=\b)")
select_normally = re.compile(r"NORMALLY")
select_mildly = re.compile(r"MILDLY")
select_moderately = re.compile(r"MODERATELY")
select_severely = re.compile(r"SEVERELY")
select_pa = re.compile(r"PULMONARY ARTERY")
select_icd_codes = re.compile(r"[A-Z](\d+\.\d*\b)")
select_slash_dates = re.compile(r"\d{2}/\d{2}/\d{4}")
select_dot_dates = re.compile(r"\d{2}\.\d{2}\.\d{4}")

space_before_unit = re.compile(r"\s+(MMHG|MM|CM|%)")
space_period = re.compile(r"\s\.")

space_plus_space = re.compile(r"\s\+\s")
verbose_pressure = re.compile(r"\+CVPMMHG")
add_period = [
    r"THE PEAK TRANSAORTIC GRADIENT IS <#>MMHG",
    r"THE MEAN TRANSAORTIC GRADIENT IS <#>MMHG",
    r"LV EJECTION FRACTION IS <#>%",
    r"ESTIMATED PA PRESSURE IS <#>MMHG",
    r"RESTING SEGMENTAL WALL MOTION ANALYSIS",
    r"THE IVC DIAMETER IS <#>MM",
    r"EST RV/RA PRESSURE GRADIENT IS <#>MMHG",
    r"ESTIMATED PEAK RVSP IS <#>MMHG",
    r"HEART FAILURE, UNSPECIFIED",
    r"CHEST PAIN, UNSPECIFIED",
    r"SINUS OF VALSALVA: <#>CM",
    r"THE PEAK TRANSMITRAL GRADIENT IS <#>MMHG",
    r"THE MEAN TRANSMITRAL GRADIENT IS <#>MMHG",
    r"ASCENDING AORTA <#>CM",
    r"ESTIMATED PA SYSTOLIC PRESSURE IS <#>MMHG",
    r"ICD_CODE SHORTNESS BREATH",
    r"ICD_CODE ABNORMAL ELECTROCARDIOGRAM ECG EKG",
    r"SHORTNESS BREATH",
    r"ABNORMAL ELECTROCARDIOGRAM ECG EKG",
    r"THE LEFT ATRIAL APPENDAGE IS NORMAL IN APPEARANCE WITH NO EVIDENCE OF THROMBUS",
]

select_number = r"(?:\d+\.?\d*)"

add_period = [re.escape(a).replace(re.escape("<#>"), select_number) for a in add_period]
add_period = [f"(?:{a})(?!\.)" for a in add_period]
add_period = "|".join(add_period)
add_period = f"({add_period})"
# print(f"{add_period[:50]} ... {add_period[-50:]}")
add_period = re.compile(add_period)


def clean_text(text):
    if len(text) > 1:
        text = text.upper()
        text = text.strip()
        text = text.replace("`", "'")
        text = removables.sub("", text)

        text = in_text_periods.sub(". ", text)
        text = square_brackets.sub("", text)

        text = select_was.sub("IS", text)
        text = select_were.sub("ARE", text)
        text = select_and_or.sub("AND", text)
        text = select_normally.sub("NORMAL", text)
        text = select_mildly.sub("MILD", text)
        text = select_moderately.sub("MODERATE", text)
        text = select_severely.sub("SEVERE", text)
        text = select_pa.sub("PA", text)
        text = select_slash_dates.sub("", text)
        text = select_dot_dates.sub("", text)
        text = select_icd_codes.sub("", text)

        text = space_before_unit.sub(r"\1", text)
        text = space_period.sub(".", text)
        text = multi_whitespace.sub(" ", text)

        text = space_plus_space.sub("+", text)
        text = verbose_pressure.sub("MMHG", text)

        text = text.strip()
        text = text + " "

        text = add_period.sub(r"\1.", text)
        text = multi_period.sub(".", text)

    return text


select_severity = "|".join(
    ["MODERATE/SEVERE", "MILD/MODERATE", "MILD", "MODERATE", "SEVERE", "VERY SEVERE"]
)
select_severity = f"((?<![A-Za-z])(?:{select_severity}))"
select_number = r"(\d+\.?\d*)"

select_variable = "|".join([select_number, select_severity])
# print(select_variable)
select_variable = re.compile(select_variable)


def extract_variables(string, replace_with="<#>"):
    matches = select_variable.findall(string)
    variables = []
    for match in matches:
        for variable in match:
            if not len(variable) == 0:
                variables.append(variable)
    variables_replaced = select_variable.sub(replace_with, string)
    return variables, variables_replaced

def fix_leakage(df, df_subset, split='train'):
    """Fix overlap in studies between the particular 'split' subset and the other subsets 
    in the dataset.
        
        Args:
        df: DataFrame of the complete dataset. 
        df_subset: A view of a subset of the DataFrame; it is either the
            train, validation or test set.
        split: Whether the df_subset is associated with the train/val/test set

        Returns:
        A dataframe of df without any data leakage problem between the train, val, test 
        subsets.
    """
    train = df[df['split']=='train']
    val = df[df['split']=='val']
    test = df[df['split']=='test']

    #Check whether any study ID in one subset is in any other subset
    val_test = val['Echo ID#'].isin(test['Echo ID#']).any()
    train_test = train['Echo ID#'].isin(test['Echo ID#']).any()
    train_val = train['Echo ID#'].isin(val['Echo ID#']).any()
    print("Checking if there is data leakage...")
    print(f"There is overlap between: val/test: {val_test}, train/test: {train_test}, train/val: {train_val}")

    #Get indices for all rows in the subset that overlap with another subset
    train_test_overlap = train['Echo ID#'].isin(test['Echo ID#'])
    train_test_leak_idx = [i for i, x in enumerate(train_test_overlap) if x]

    val_test_overlap = val['Echo ID#'].isin(test['Echo ID#'])
    val_test_leak_idx = [i for i, x in enumerate(val_test_overlap) if x]
    
    #Get unique study IDs corresponding to the overlapping rows
    train_test_leak_ids = train['Echo ID#'].iloc[train_test_leak_idx].to_list()
    train_test_leak_ids = list(set(train_test_leak_ids))

    val_test_leak_ids = val['Echo ID#'].iloc[val_test_leak_idx].to_list()
    val_test_leak_ids = list(set(val_test_leak_ids))

    print(f"Echo IDs of overlapping studies between: val/test: {val_test_leak_ids}, train/test: {train_test_leak_ids}")

    #Assign overlapping studies to only one subset
    num_remove_test = len(train_test_leak_ids)//2
    remove_test_ids = train_test_leak_ids[0:num_remove_test]
    remove_train_ids = train_test_leak_ids[num_remove_test:]  

    num_remove_val = len(val_test_leak_ids)//2
    remove_val_ids = val_test_leak_ids[0:num_remove_val]
    remove_test_ids = remove_test_ids + val_test_leak_ids[num_remove_val:]  

    if split == 'train':
        fixed_subset = remove_ids(remove_ids=remove_train_ids, dataset=df_subset)
        if len(fixed_subset) == len(df_subset) - 5:
            print("Data leakage for train/test subsets has been fixed.")
    elif split == 'val':
        fixed_subset = remove_ids(remove_ids=remove_val_ids, dataset=df_subset)
    elif split == 'test':  
        fixed_subset = remove_ids(remove_ids=remove_test_ids, dataset=df_subset)
        if len(fixed_subset) == len(df_subset) - 8:
            print("Data leakage for train/test subsets has been fixed.")
    
    return fixed_subset

def remove_ids(remove_ids, dataset):
    "Remove rows with 'Echo ID#' in the list of remove_ids for the dataset"
    for id in remove_ids:
        remove_rows = dataset[dataset['Echo ID#']==id].index.values
        dataset = dataset.drop(index=remove_rows)
    
    return dataset

    
    



#utils from tabular transformer finetuning branch
def preprocess_as_data(train, val, test, cat_cols):

    #train = train.replace(-1, np.nan)
    #val = val.replace(-1, np.nan)
    #test = test.replace(-1, np.nan)
    
    # Remove non numerical features
    numeric_feats = train.columns.to_list().copy()
    numeric_feats = [col for col in numeric_feats if col not in cat_cols]
    numeric_feats.remove('as_label')

    # Replace any missing values in the categorical columns with -1
    train[cat_cols] = train[cat_cols].fillna(-1)
    val[cat_cols] = val[cat_cols].fillna(-1)
    test[cat_cols] = test[cat_cols].fillna(-1) 

    processor = make_column_transformer(
        (GaussianImputerGivenMean(strategy='mean', mean=1.5, std=0.3), ['VPeak']),
        remainder='passthrough'
    )

    # Create a new list without 'VPeak' - this is a current hack for the imputation and can be cleaned up
    new_numeric_feats = [feat for feat in numeric_feats if feat != 'VPeak']
    all_columns = ['VPeak'] + new_numeric_feats + cat_cols

    pipe = make_pipeline(StandardScaler(), GaussianImputer())
    processor2 = make_column_transformer(
        (pipe, numeric_feats),
        remainder='passthrough'
    )

    # get normalized versions and PeakV of each set for numeric features only
    train_temp = pd.DataFrame(processor.fit_transform(train.drop(columns=['as_label'])), columns=all_columns, index=train.index)
    val_temp = pd.DataFrame(processor.transform(val.drop(columns=['as_label'])), columns=all_columns, index=val.index)
    test_temp = pd.DataFrame(processor.transform(test.drop(columns=['as_label'])), columns=all_columns, index=test.index)

    # get imputed versions of each set for numeric features only
    # imputation for Peak gradient
    train_temp = fill_peak_gradient(train_temp)
    val_temp = fill_peak_gradient(val_temp)
    test_temp = fill_peak_gradient(test_temp)

    #iterative imputation for the remaining columns?
    train_impute = pd.DataFrame(processor2.fit_transform(train_temp), columns=all_columns, index=train.index)
    val_impute = pd.DataFrame(processor2.transform(val_temp), columns=all_columns, index=val.index)
    test_impute = pd.DataFrame(processor2.transform(test_temp), columns=all_columns, index=test.index)

    # train = train.replace(np.nan, -1)
    # val = val.replace(np.nan, -1)
    # test = test.replace(np.nan, -1)

    # create a dataset with each of these
    # train_set = ASDataset(train, train_impute, all_columns)
    # val_set = ASDataset(val, val_impute, all_columns)
    # test_set = ASDataset(test, test_impute, all_columns)

    return (train_impute, val_impute, test_impute, all_columns)

def load_as_data(csv_path: str,
                    drop_cols : Union[List[str], None] = None,
                    num_ex : Union[int, None] = None,
                    test_split : float = 0.1,
                    random_seed : Union[int, None] = None,
                    scale_feats : bool = True
                    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
        """Processes data for imputation models.

        Imputes missing values with average of feature and optionally drops columns and scales all 
        data to be normally distributed.
        
        Args:
            csv_path: Path to the dataset to use. Should be a csv file.
            drop_cols: List of columns to drop from dataset. If None, no columns are dropped.
            num_ex: Number of examples to use. If None, will use all available examples.
            test_split: What fraction of total data to use in test set. Also used to split
                validation data after test data has been separated.
            random_seed: Seed to initialize randomized operations.
            scale_feats: Whether to scale numeric features in dataset during preprocessing.

        Returns:
            Tuple of (train_dataset, validation_dataset, test_dataset).
            Each is a processed pandas DataFrame.

        Raises:
            Exception: Specified more examples to use than exist in the dataset.
        """

        data_df = pd.read_csv(csv_path, index_col=0)
        data_df = data_df.drop("AV stenosis", axis=1)
        #data_df = data_df.drop("age", axis=1)

        #If num_ex is None use all examples in dataset
        if not num_ex:
            num_ex = data_df.shape[0]

        #Ensure number of examples specified is not greater than examples in the dataset
        elif num_ex > data_df.shape[0]:
            ex_string = "Specified " + str(num_ex) + " examples to use but there are only " + str(nan_df.shape[0]) + " examples with known target column in dataset."
            raise Exception(ex_string)
        

        #Create description of processing and store
        print("Processing data from: " + csv_path + "\n")
        print("Dropping the following columns: " + str(drop_cols) + "\n")
        print("Using " + str(num_ex) + " examples with test split of " + str(test_split) + ".\n")
        print("Random seed is " + str(random_seed) + ".\n")
        print("Scaling features? " + str(scale_feats) + "\n")

        #Replace any -1 values with NaNs for imputing
        #nan_df = data_df.replace(-1, np.nan)
        
        #Sample data to only contain num_ex rows
        sampled_df = data_df.sample(n=num_ex, random_state=random_seed)

        #If drop columns is not empty, drop specified. Otherwise keep DataFrame as is
        drop_cols_df = sampled_df.drop(columns=drop_cols) if drop_cols else sampled_df

        #Split into train, test, and validation sets
        train_df = drop_cols_df[drop_cols_df['split'] == 'train'].drop(columns=['split'])
        val_df = drop_cols_df[drop_cols_df['split'] == 'val'].drop(columns=['split'])
        test_df = drop_cols_df[drop_cols_df['split'] == 'test'].drop(columns=['split'])

        print("\nTrain dataset shape:", train_df.shape)
        print("Validation dataset shape:", val_df.shape)
        print("Test dataset shape:", test_df.shape,"\n")

        return (train_df, val_df, test_df)

def fill_peak_gradient(df):
    # Find rows where 'AoPG' is NaN
    missing_ao_rows = df['AoPG'].isna()

    # Replace NaN values in 'AoPG' with the calculated values based on the formula
    df.loc[missing_ao_rows, 'AoPG'] = 4 * (df.loc[missing_ao_rows, 'VPeak']**2)

    return df

class GaussianImputer(TransformerMixin):
    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y=None):
        self.means_ = np.nanmean(X, axis=0)
        self.stddevs_ = np.nanstd(X, axis=0)
        return self

    def transform(self, X):
        X = check_array(X, force_all_finite=False)

        for i in range(X.shape[1]):
            nan_mask = np.isnan(X[:, i])
            num_missing = np.sum(nan_mask)
            if num_missing > 0:
                random_values = np.random.normal(loc=self.means_[i], scale=self.stddevs_[i], size=num_missing)
                X[nan_mask, i] = random_values

        return X
    
class GaussianImputerGivenMean(SimpleImputer):
    def __init__(self, strategy='mean', fill_value=None, mean=None, std=None, **kwargs):
        self.mean = mean
        self.std = std
        super().__init__(strategy=strategy, fill_value=fill_value, **kwargs)

    def transform(self, X):

        # Get the indices of missing values in the column
        missing_indices = np.where(np.isnan(X))[0]

        # Generate random samples from a Gaussian distribution around mean=1.5
        fill_values = np.random.normal(loc=self.mean, scale=self.std, size=len(missing_indices))

        # Clip values to ensure they are not less than 0
        fill_values = np.clip(fill_values, 0, None)

        # Reshape the fill_values array to be a column vector
        fill_values = fill_values.reshape(-1, 1)

        # Create a new DataFrame and replace the original one
        new_X = X.copy()
        new_X.iloc[missing_indices] = fill_values

        return new_X

def update_confusion_matrix(mat, true, pred):
    '''
    updates a confusion matrix in numpy based on target/pred results
    with true results based on the row index, and predicted based on column

    Parameters
    ----------
    mat : int
        NxN confusion matrix
    true : Bx1 List[int]
        list of integer labels
    pred : Bx1 List[int]
        list of integer labels

    Returns
    -------
    NxN confusion matrix
    '''
    for i in range(len(true)):
        mat[true[i],pred[i]] = mat[true[i],pred[i]] + 1
    return mat

def acc_from_confusion_matrix(mat):
    # get accuracy from NxN confusion matrix
    return np.trace(mat)/np.sum(mat)