data{
  int<lower=1> Nsubject;    // number of subjects
  int<lower=1> Ntime;    // total number of time-steps
  
  int<lower=1> Nobs; // length of observed data
  real<lower=0, upper=5> MAL_obs[Nobs]; // flattened observed MAL data 
  int<lower=1, upper=Nsubject> id_obs[Nobs]; // subject indices for MAL_obs
  int<lower=1, upper=Ntime> t_obs[Nobs]; // time indices for MAL_obs
  
  // boolean matrix (stored as real) for whether training occured
  matrix[Ntime, Nsubject] train_time; 
  matrix[Ntime, Nsubject] train_dose; // dose amount for each week and user.
  row_vector<lower=0, upper=5>[Nsubject] MAL_init; // observed MAL when t=0
  real MAL_pred_slope; // meta-parameter: slope of sigmoid output
}


transformed data{
  matrix[Ntime, Nsubject] not_train_time = 1 - train_time;
  
}

parameters{
  // subject independent parameters: measurement noise
  real<lower=0> MAL_sd;
  real<lower=0> nu; //degree of freedom for likelihood (for MAL outliers)
  real<lower=0> x_init_sd;
  
  // Level 1: Population; hyper parameters
  real mu_alpha; 
  real mu_beta;  
  real mu_gamma;
  real<lower=0> sd_alpha; 
  real<lower=0> sd_beta;
  real<lower=0> sd_gamma; 

  // Level 2: covariates
  real<lower=0> slope_MALinit_xinit;
  
  // Level 3: Subjects; individual parameters
  row_vector<lower=0>[Nsubject] x_init;
  row_vector<lower=0>[Nsubject] beta;
  row_vector<lower=0>[Nsubject] gamma; 

  row_vector[Nsubject] RE_alpha;

}

transformed parameters{
  row_vector<lower=0, upper=1>[Nsubject] alpha = inv_logit(mu_alpha + RE_alpha); 

  // hidden states
  matrix<lower=0, upper=5>[Ntime, Nsubject] MAL; // complete data
  matrix<lower=0>[Ntime, Nsubject] x;

  // first "true" MAL and memory state 
  x[1] =  alpha .* x_init +  gamma .* MAL_init; 
  MAL[1] =  10 * inv_logit(MAL_pred_slope * x[1]) - 5; 

  for(t in 2:Ntime){
    // memory state for subject s after t weeks.
    x[t] = alpha .* x[t-1]  + beta .* train_dose[t-1] + not_train_time[t-1] .* gamma .* MAL[t-1]; 
    MAL[t] = 10 * inv_logit(MAL_pred_slope * x[t]) - 5;
  }
}

model{
  // hyperpriors
  mu_alpha ~ normal(2, 1); 
  sd_alpha ~ inv_gamma(3,2); 
  
  mu_beta ~ normal(0, 1); 
  sd_beta ~ inv_gamma(4,2); 
  
  mu_gamma ~ normal(0, 1);
  sd_gamma ~ inv_gamma(4,2); 

  
  // subject independent priors: measurement noise
  MAL_sd ~ normal(0.25, 0.1);
  x_init_sd ~ inv_gamma(3,2);
  nu ~  gamma(2,0.1);
  
  // covariates
  slope_MALinit_xinit ~ normal(0, 2);
  
  // inidivudal priors
  RE_alpha ~ normal(0, sd_alpha);
  beta ~ normal(mu_beta, sd_beta); 
  gamma ~ normal(mu_gamma, sd_gamma);

  x_init ~ normal(slope_MALinit_xinit * MAL_init, x_init_sd); // Initialization of the first memory
  
  for (i in 1:Nobs){
    MAL_obs[i] ~ student_t(nu, MAL[t_obs[i], id_obs[i]], MAL_sd);
  }
}


generated quantities{
  vector[Nobs] log_lik; // flattened Log likelihood for MAL_measured
  for (i in 1:Nobs){
    log_lik[i] = student_t_lpdf(MAL_obs[i] | nu, MAL[t_obs[i], id_obs[i]], MAL_sd);
  }
  
}








