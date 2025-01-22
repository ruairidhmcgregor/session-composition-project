import pandas as pd
import scipy.stats as st

for i, cols in enumerate(residuals_cols):
    
    # Initialize lists to store means and confidence intervals
    means = []
    ci95_lows = []
    ci95_highs = []
    errors = []
    st_devs = []
    
    # Calculate mean and confidence intervals for each column in the list
    for col in cols:
        mean = df[col].mean()
        st_dev = df[col].std()
        
        std_err = st.sem(df[col])
        n = len(df[col])
        ci95_low = st.norm.interval(confidence=0.95, loc=mean, scale=st.sem(df[col]))[0]
        ci95_high = st.norm.interval(confidence=0.95, loc=mean, scale=st.sem(df[col]))[1]
        
        means.append(mean)
        st_devs.append(st_dev)
        ci95_lows.append(ci95_low)
        ci95_highs.append(ci95_high)
        errors.append(std_err)
