import numpy as np

def aeroelastic_scaling(k_l,T_scaled=300.0,T_full=227.0,mach=0.85,gamma=1.4):
  """
  k_l : length scale factor (k_l=L_scaled/L_full)
  k_t : time scale factor (k_t=T_scaled/T_full)
  k_m : mass scale factor (k_m=M_scaled/M_full)
  k_f : force scale factor (k_f=F_scaled/F_full)
  """
  k_v=np.sqrt(T_scaled/(1.0+(gamma-1.0)/2.0*mach**2)/T_full)
  k_t=k_l/k_v
  k_rho=1/k_v/k_l
  k_m=k_l**3*k_rho
  k_f=k_m*k_l/k_t**2
  return k_t,k_m,k_f

def aeroelastic_scaling_wt(k_l,k_p,T_scaled=300.0,T_full=227.0,mach=0.85,gamma=1.4):
  """
  k_l : length scale factor (k_l=L_scaled/L_full)
  k_p : pressure scale factor (k_p=P_scaled/P_full)
  k_t : time scale factor (k_t=T_scaled/T_full)
  k_m : mass scale factor (k_m=M_scaled/M_full)
  k_f : force scale factor (k_f=F_scaled/F_full)
  """
  k_v=np.sqrt(T_scaled/(1.0+(gamma-1.0)/2.0*mach**2)/T_full)
  k_f=k_p*k_l**2
  k_t=k_l/k_v
  k_m=k_f*k_t**2/k_l
  return k_t,k_m,k_f
