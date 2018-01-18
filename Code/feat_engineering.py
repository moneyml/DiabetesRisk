# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 13:35:17 2017

@author: maoli
"""

import pandas as pd
import numpy as np
import random



def count_single_col(dataset,var_list,keep_list = []):
    for var in var_list:
        if not var in dataset.columns:
            continue
        dataset['cnt_'+var] = 0
        keep_list.append('cnt_'+var)
        groups = dataset.groupby([var])
        for name,group in groups:
            count = group[var].count()
            dataset['cnt_'+var].ix[group.index] = count
    return dataset[keep_list]


def count_by_other_col(dataset,var_dict,keep_list = []):
    for tool,pro in var_dict.items():
        try:
            groups = dataset.groupby(['TOOL_'+tool[-3:]])
        except:
            continue
        for var in pro:
            if not var in dataset.columns:
                continue
            dataset['cnt_'+tool+'_'+var] = 0
            keep_list.append('cnt_'+tool+'_'+var)
            for name,group in groups:
                grps = group.groupby([var])
                for name2,grp in grps:
                    dataset['cnt_'+tool+'_'+var].ix[grp.index] = float(len(grp))/float(len(group))
    return dataset[keep_list]



def pcent_single_col(dataset,var_list,keep_list = []):
    for var in var_list:
        if not var in dataset.columns:
            continue
        if 'TOOL' not in var:
            dataset['pcent_'+var] = dataset[var].rank(method='max')/float(len(dataset))
            keep_list.append('pcent_'+var)
    return dataset[keep_list]


def pcent_by_other_col(dataset,var_dict,keep_list = []):
    for tool,pro in var_dict.items():
        try:
            groups = dataset.groupby(['TOOL_'+tool[-3:]])
        except:
            continue
        for var in pro:
            if not var in dataset.columns:
                continue
            dataset['pcent_'+tool+'_'+var] = 0
            keep_list.append('pcent_'+tool+'_'+var)
            for name,group in groups:
                dataset['pcent_'+tool+'_'+var].ix[group.index] = group[var].rank(method='max')/float(len(group))
    return dataset[keep_list]



def var_minus(dataset,var_list,keep_list = [],criteria=0.8):
    for i in range(len(var_list)-1):
        var1_unique = dataset[var_list[i]].unique().tolist()
        for j in range(i+1,len(var_list)):
            var2_unique = dataset[var_list[j]].unique().tolist()
            combine_list = set(var1_unique+var2_unique)
            if float(len(var1_unique))/len(combine_list)>=criteria and float(len(var2_unique))/len(combine_list)>=criteria:
                tmp_var = var_list[i]+'-'+var_list[j]
                dataset[tmp_var] = dataset[var_list[i]]-dataset[var_list[j]]
                keep_list.append(tmp_var)
    return dataset[keep_list]


'''def sim_group(dataset,var_list):
    base_list = var_list.copy()
    range_df = pd.DataFrame({'min':dataset[var_list].min(),'max':dataset[var_list].max()},index=var_list)
    range_df.sort_values(['max','min'],inplace=True)
    range_df['range'] = 1
    tmp = range_df.groupby(['max','min'])['range'].sum()
    tmp.reset_index(inplace=True) 
    range_dict ={}
    for i in tmp.index:
        tmp_list = range_df.loc[(range_df['min']==tmp.loc[i,'min'])&(range_df['max']==tmp.loc[i,'max'])].index.tolist()
        range_dict[tmp_list] = []
        for var in tmp_list:
            
    for var_list in range_list:
        for var in var_list:
            base_list.remove(var)'''
    
    



















