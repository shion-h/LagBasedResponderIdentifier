functions {//{{{
    //
    int[] return_beta_type(int cross_over_type){
        int beta_type[5];
        beta_type[1] = 1;
        beta_type[2] = 2 + (cross_over_type - 1);
        beta_type[3] = 1;
        beta_type[4] = 2 + (2 - cross_over_type);
        beta_type[5] = 1;
        return beta_type;
    }

    real[] return_beta(real alpha, real eta, real eta_tilde){
        real beta[3];
        beta[1] = exp(alpha);
        beta[2] = exp(alpha + eta_tilde);
        beta[3] = exp(alpha + eta_tilde + eta);
        return beta;
    }

    int[] return_d_prime(int[] d, int N, int mu, int nu){
        int d_prime[4];
        d_prime[1] = d[1] + mu;
        d_prime[2] = d[2] + nu;
        d_prime[3] = d[3] + mu;
        d_prime[4] = d[4] + nu;
        for (b in 1:4){
            if (d_prime[b] > N + 1){
                d_prime[b] = N + 1;
            }
        }
        return d_prime;
    }

    real[] return_gamma(int[] y, int[] d_prime, real[] beta, int[] beta_type){
        real gamma[5];
        gamma[1] = 0;
        for (t in 2:5){
            // y[s, d[b] - 1]を通るように
            gamma[t] = y[d_prime[t-1] - 1] - beta[beta_type[t]] * (d_prime[t-1] - 1);
        }
        return gamma;
    }

    real return_mean_y(int[] t, int[] d_prime, real[] beta, int[] beta_type, int i, real[] gamma){
        int rgr_index;
        real mean_y;
        if(t[i] < d_prime[1]){
            rgr_index = 1;
        }else if(t[i] < d_prime[2]){
            rgr_index = 2;
        }else if(t[i] < d_prime[3]){
            rgr_index = 3;
        }else if(t[i] < d_prime[4]){
            rgr_index = 4;
        }else{
            rgr_index = 5;
        }
        mean_y = beta[beta_type[rgr_index]] * i + gamma[rgr_index];
        return mean_y;
    }

}//}}}

data {//{{{
    int<lower=0> N;
    int<lower=0> y[N];
    int<lower=0> t[N];
    int<lower=0> mu_max;
    int<lower=0> nu_max;
    int<lower=0> d[4];
    int<lower=1> cross_over_type;
    real<lower=0> sigma_y_lowerbound;
}//}}}

transformed data {//{{{
}//}}}

parameters {//{{{
    // logarithmic defecation frequency of normal periods
    real alpha;
    // effect of target probiotics
    real eta;
    // effect of capsules
    real eta_tilde;
    real<lower=sigma_y_lowerbound> sigma_y;
}//}}}

transformed parameters {//{{{
    real beta[3];
    matrix[mu_max+1, nu_max+1] mu_nu_prob;
    real log_sum_exp_mu_nu_prob;

    beta = return_beta(alpha, eta, eta_tilde);
    for (mu in 0:mu_max){
        for (nu in 0:nu_max){
            mu_nu_prob[mu + 1, nu + 1] = 0;
        }
    }
    // cross_over_type: 1 means placebo -> test 2 means test -> placebo
    // set a block to avoid being recognized as integer parameters
    {
    int beta_type[5] = return_beta_type(cross_over_type);
    for (mu in 0:mu_max){
        for (nu in 0:nu_max){
            int d_prime[4] = return_d_prime(d, N, mu, nu);
            // bias term
            real gamma[5] = return_gamma(y, d_prime, beta, beta_type);

            for (i in 1:N){
                real tmp_log_prob = 0;
                // beta_i i + gamma_i
                real mean_y = return_mean_y(t, d_prime, beta, beta_type, i, gamma);
                tmp_log_prob = normal_lpdf(y[i] | mean_y, sigma_y);

                mu_nu_prob[mu + 1, nu + 1] += tmp_log_prob;
            }
        }
    }
    log_sum_exp_mu_nu_prob = log_sum_exp(mu_nu_prob);
    }
}//}}}

model {//{{{
    target += log_sum_exp_mu_nu_prob;
    alpha ~ cauchy(0, 10);
    eta ~ cauchy(0, 10);
    eta_tilde ~ cauchy(0, 10);
    sigma_y ~ cauchy(0, 10);
}//}}}

generated quantities {//{{{
    matrix[mu_max+1, nu_max+1] mu_nu_posterior;
    real y_pred[N];
    // compute posterior distributions of mu and nu
    for (mu in 0:mu_max){
        for (nu in 0:nu_max){
            mu_nu_posterior[mu + 1, nu + 1] = mu_nu_prob[mu + 1, nu + 1] - log_sum_exp_mu_nu_prob;
        }
    }
    // draw y sample
    {
    int beta_type[5] = return_beta_type(cross_over_type);
    int mu = categorical_rng(rep_vector(1.0/(mu_max+1), mu_max+1)) - 1;
    int nu = categorical_rng(rep_vector(1.0/(nu_max+1), nu_max+1)) - 1;
    int d_prime[4] = return_d_prime(d, N, mu, nu);
    real gamma[5] = return_gamma(y, d_prime, beta, beta_type);

    for (i in 1:N){
        int rgr_index;
        real tmp_log_prob = 0;
        real mean_y = return_mean_y(t, d_prime, beta, beta_type, i, gamma);
        y_pred[i] = normal_rng(mean_y, sigma_y);
    }
    }
}//}}}
