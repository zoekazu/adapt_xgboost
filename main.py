#!/usr/bin/env python3
# -*- Coding:utf-8 -*-
# %%
import os
from src.utils import confirm_make_folder
from IPython.display import display, Image
import numpy as np
import cv2
from src.read_dir_images import ImgsInDirAsBool
import pandas as pd
# %%
df_xgboost = pd.read_pickle('./pandas_df_connected_xgboost.pkl')
df_xgboost.head()

# %%
true_files = ImgsInDirAsBool('./images/hard/true', bool_switch=True)
false_files = ImgsInDirAsBool('./images/hard/false', bool_switch=True)

# %%


def display_cv(image, format='.bmp', bool_switch=False):
    if bool_switch:
        image = image.astype(np.uint8)*255
    decoded_bytes = cv2.imencode(format, image)[1].tobytes()
    display(Image(data=decoded_bytes))


# %%
output_folder = 'result'
confirm_make_folder(output_folder)
# %%
for num, (true, false, img_name) in enumerate(
        zip(true_files.read_files(),
            false_files.read_files(),
            true_files.files_name()),
        start=1):
    df_xgboost_part = df_xgboost[df_xgboost['image_No'] == num].copy()
    true_or_false = np.logical_or(true, false)
    nlabels, labels, labels_status, center_object = cv2.connectedComponentsWithStats(
        true_or_false.astype(np.uint8)*255, connectivity=8)
    xgboost_adapt_bools = np.zeros([labels.shape[0], labels.shape[1], nlabels - 1], dtype=bool)
    xgboost_adapt_bool = np.zeros([labels.shape[0], labels.shape[1]], dtype=bool)
    for i in range(1, nlabels):
        xgboost_adapt_bools[:, :, i-1] = np.where(labels == i, True, False)
    for i in range(nlabels-1):
        if df_xgboost_part['xgboost_result'].iloc[i] == False:
            xgboost_adapt_bool = np.logical_or(xgboost_adapt_bool, xgboost_adapt_bools[:, :, i])
    # display_cv(true_or_false, bool_switch=True)
    true_or_false = np.logical_and(true_or_false, np.logical_not(xgboost_adapt_bool))
    # display_cv(true_or_false, bool_switch=True)
    image_name = os.path.splitext(os.path.basename(img_name))[0]
    outim_name = './{0}/{1}_xgboost.bmp'.format(output_folder, image_name[:-5])
    cv2.imwrite(outim_name, np.logical_not(true_or_false).astype(np.uint8)*255)

# %%
