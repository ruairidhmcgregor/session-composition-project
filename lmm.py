import pandas as pd
import statsmodels.formula.api as smf

def lmm(var_string, metric, group_col):

    #fit the model
    mixed = smf.mixedlm(metric + " ~ " + var_string, df, groups = df[group_col])
    mixed_fit = mixed.fit()
    #print the summary
    print(mixed_fit.summary())

    result = mixed_fit

    var_resid = result.scale
    var_random_effect = float(result.cov_re.iloc[0])
    var_fixed_effect = result.predict(df).var()

    total_var = var_fixed_effect + var_random_effect + var_resid
    marginal_r2 = var_fixed_effect / total_var
    conditional_r2 = (var_fixed_effect + var_random_effect) / total_var
    
    return conditional_r2
