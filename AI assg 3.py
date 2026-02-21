#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Step 1: Load the three CSV files
trail1 = pd.read_csv("Trail1_extracted_features_acceleration_m1ai1-1.csv")
trail2 = pd.read_csv("Trail2_extracted_features_acceleration_m1ai1.csv")
trail3 = pd.read_csv("Trail3_extracted_features_acceleration_m2ai0.csv")


# In[13]:


# Step 2: Combine into a single dataset
df = pd.concat([trail1, trail2, trail3], ignore_index=True)
print(f"Combined dataset shape: {df.shape}")
print(f"Event distribution:\n{df['event'].value_counts(dropna=False)}")


# In[15]:


# Step 3: Remove unnecessary columns
columns_to_remove = ['start_time', 'axle', 'cluster', 'tsne_1', 'tsne_2']
# Use errors='ignore' because not all files have cluster/tsne columns
df = df.drop(columns=columns_to_remove, errors='ignore')


# In[16]:


# Step 4: Encode the 'event' column (binary classification)
# 'normal' -> 0, everything else -> 1
df['event'] = np.where(df['event'].astype(str).str.strip().str.lower() == 'normal', 0, 1)

print(f"Class distribution after encoding:")
print(f"  Normal (0): {(df['event'] == 0).sum()}")
print(f"  Event  (1): {(df['event'] == 1).sum()}")


# In[17]:


# Step 5: Separate features and target
X = df.drop(columns=['event'])
y = df['event']

# Ensure all features are numeric (safe for StandardScaler)
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)


# In[18]:


# Normalize features using StandardScaler (zero mean, unit variance)
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns
)

print(f"Before scaling - Mean: {X.mean().mean():.4f}, Std: {X.std().mean():.4f}")
print(f"After scaling  - Mean: {X_scaled.mean().mean():.4f}, Std: {X_scaled.std().mean():.4f}")


# In[ ]:





# In[ ]:




