functions {//{{{
}//}}}

data {//{{{
    int<lower=0> N;
    int<lower=0> D;
    real<lower=0,upper=1> y[N];
    vector[D] x[N];
}//}}}

transformed data {//{{{
}//}}}

parameters {//{{{
    vector[D] beta;
    /*
    real<lower=0, upper=pi()/2> phi_unif;
    real<lower=0, upper=pi()/2> lambda_unif;
    */
    real<lower=0> phi;
    real<lower=0> lambda;
}//}}}

transformed parameters {//{{{
    real theta[N];
    /*
    real phi;
    real lambda;
    phi = 0 + 10 * tan(phi_unif);
    lambda = 0 + 10 * tan(lambda_unif);
    */
    for (i in 1:N){
        /*
        real expbetax;
        expbetax = exp(beta' * x[i]);
        theta[i] = expbetax / (1 + expbetax);
        */
        real betax;
        betax = beta' * x[i];
        theta[i] = inv_logit(betax);

    }
}//}}}

model {//{{{
    phi ~ cauchy(0, 10);
    lambda ~ cauchy(0, 10);
    beta ~ normal(0, lambda);
    for (i in 1:N){
        y[i] ~ beta(theta[i] * phi, (1 - theta[i]) * phi);
        // y[i] ~ beta(1, 1);
    }
}//}}}

generated quantities {//{{{
}//}}}
