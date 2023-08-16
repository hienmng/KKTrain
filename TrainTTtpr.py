#!/usr/bin/env python
# coding: utf-8

# # Training File for Trigger Dataset

# This is a training file for the Trigger dataset (cluster number 60358177) located at "/global/cfs/cdirs/m3712/Mu2e/TrkAna/60358177". Running the following cells will call and run TrainBkg.ipynb. The AUC curve generated will be saved as "TrainBkgTTtpr.pdf" in the "training_plots" directory, and the .h5 file training model generated will be saved as "TrainBkgTTtpr.h5" in the "models" file

# In[1]:


## This cell runs on the tpr tree

suffix = "TTtpr"
treename = "TAtpr"
file_list = "/global/cfs/cdirs/m3712/Mu2e/TrkAna/60358177/files.txt"
#file_list = "/Users/brownd/data/60358177/all_files.txt"
print("Using files in " + file_list)

get_ipython().run_line_magic('run', './TrainBkg.ipynb')


# In[2]:


## This cell runs on the cpr tree

suffix = "TTcpr"
treename = "TAcpr"
file_list = "/global/cfs/cdirs/m3712/Mu2e/TrkAna/60358177/files.txt"
##file_list = "/Users/brownd/data/60358177/all_files.txt"
print("Using files in " + file_list)

get_ipython().run_line_magic('run', './TrainBkg.ipynb')

