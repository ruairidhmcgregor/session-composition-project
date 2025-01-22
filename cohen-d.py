def pooled_standard_deviation(n1, s1, n2, s2):
  """
  Calculate pooled standard deviation for Cohen's d calculation.
  
  Parameters:
      n1 (int): Sample size of group 1.
      s1 (float): Standard deviation of group 1.
      n2 (int): Sample size of group 2.
      s2 (float): Standard deviation of group 2.
  
  Returns:
      float: Pooled standard deviation.
  """
  return ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)

def cohen_d(cat1_mean, cat2_mean, cat1_sd, cat2_sd, cat1_sample, cat2_sample):
  """
  Calculate Cohen's d using pooled standard deviation.
  
  Parameters:
      cat1_mean (float): Mean of group 1.
      cat2_mean (float): Mean of group 2.
      cat1_sd (float): Standard deviation of group 1.
      cat2_sd (float): Standard deviation of group 2.
      cat1_sample (int): Sample size of group 1.
      cat2_sample (int): Sample size of group 2.
  
  Returns:
      float: Cohen's d effect size.
  """
  # Calculate pooled standard deviation
  sp = pooled_standard_deviation(cat1_sample, cat1_sd, cat2_sample, cat2_sd)
  
  # Calculate Cohen's d
  return (cat1_mean - cat2_mean) / sp
