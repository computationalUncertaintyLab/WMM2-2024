
#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.integrate import solve_ivp
from joblib import Parallel, delayed

import pickle

if __name__ == "__main__":

    time_series_hour = pd.read_csv("./analysis_data/time_series_hourscale.csv")
    time_series_day  = pd.read_csv("./analysis_data/time_series_dayscale.csv")


    def model_framework(y, N):
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        from numpyro.infer.autoguide import AutoNormal, AutoLowRankMultivariateNormal, AutoDelta
        from numpyro.optim import Adam
        from numpyro.infer import SVI, Trace_ELBO, init_to_median

        from   numpyro.infer import Predictive


        y+=0.1
        eps                = 10**-5

        pop                = 8000
        max_cases_at_start = jnp.max(y[:,0])

        def model(y,N, predict=False):
            from diffrax import diffeqsolve, ODETerm, Dopri5, Heun, SaveAt
            import jax.numpy as jnp
            from jax.scipy.special import logsumexp

            def f(t, y, args):
                S,I,i        = y
                repo, gamma, N = args

                repo_timed = jnp.where((t % 24 > 7) , repo, 0)
                beta       = repo_timed*gamma

                dS = -beta*S*I/N
                dI =  beta*S*I/N 
                di =  beta*S*I/N 
                return jnp.array([dS,dI,di])

            seasons, weeks =  y.shape

            season_strata = numpyro.plate("strata", 2, dim=-2)

            #--priors-------------------------------------------------------
            I0 = numpyro.sample( "I0", dist.Beta(1,10) )
            I0 = max_cases_at_start*I0

            infectious_period = 12 #numpyro.sample( "gamma", dist.Gamma(1,1) ) 
            gamma             = 1./infectious_period

            sus      = numpyro.sample("sus", dist.Beta(1,1))

            repo     = numpyro.sample("repo"    , dist.Beta(1,1))
            repo     = 5*(repo+1)
           
            #---------------------------------------------------------------
            dt       = 1.
            saves    = SaveAt(ts = jnp.arange(-1.,weeks,1) )

            term     = ODETerm(f)
            solver   = Heun()
            y0       = jnp.array([sus*pop-I0,I0,I0])

            def run_ode(repo,gamma,pop):
                sol = diffeqsolve(term
                                  , solver
                                  ,   t0     = -1
                                  ,   t1     = weeks+1
                                  ,   dt0    = dt
                                  ,   y0     = y0
                                  ,   saveat = saves
                                  ,   args   = (repo,gamma,pop))
                return sol
            solution = run_ode(repo, gamma, pop)

            time_units = solution.ts[1:]
            cinc       = solution.ys[:,-1]

            inc        = jnp.diff(cinc, axis=0)

            inc_prop   = numpyro.deterministic("inc_prop", inc/pop)
            inc        = numpyro.deterministic("inc", inc)

            with numpyro.plate("seasons", seasons, dim=-2 ):
                with numpyro.plate("times", weeks, dim=-1 ):
                    present = ~jnp.isnan(y)
                    numpyro.sample( "loglik"
                                    #, dist.NegativeBinomial2( inc.reshape(1, weeks) + 0.1, 10 )
                                    , dist.Poisson( inc.reshape(1, weeks) + 0.1 )
                                    , obs = y.reshape(seasons, weeks) )
            if predict:
                numpyro.sample("predict", dist.Poisson(inc.reshape(1, weeks)))
                #numpyro.sample("predict", dist.NegativeBinomial2(inc.reshape(1, weeks), 10))
                

        guide = AutoDelta( model, init_loc_fn=init_to_median)

        optimizer  = numpyro.optim.Adam(step_size=0.001)
        svi        = SVI(model, guide, optimizer, loss=Trace_ELBO())
        svi_result = svi.run(jax.random.PRNGKey(0)
                             , 5*10**3
                             , y     = y
                             , N     = N)

        params_map = guide.median(svi_result.params)

        from numpyro.infer.util import init_to_value
        init_fn = init_to_value(values=params_map)

        mcmc = MCMC(NUTS(model, max_tree_depth=4, init_strategy = init_fn), num_warmup=8*10**3, num_samples=10*10**3)
        mcmc.run(jax.random.PRNGKey(1)
                 , y             = y
                 , N             = N)
        mcmc.print_summary()
        samples = mcmc.get_samples()

        # Generate posterior predictive samples using previously drawn MCMC samples
        from numpyro.infer import Predictive

        # Define model as used in control_fit (reusing trace)
        predictive = Predictive(model
                                ,posterior_samples=samples
                                ,return_sites=["predict"])

        preds = predictive(jax.random.PRNGKey(2)
                           ,y        = y
                           , N = N
                           ,predict = True)

        yhats = preds["predict"]

        return samples, yhats
        
    analysis_data = time_series_hour.loc[~time_series_hour.dow.isin([5,6])]
    y = pd.pivot_table( index=["week","dow"], columns = ["hour"], values = ["inc"], data = analysis_data )
    y.columns = [y for x,y in y.columns ]

    #change the below back to zero.
    y         = y[ list(np.arange(0,24)) ]
    y         = y.to_numpy()
    
    samples,yhats = model_framework(y=y,N=8000)

    pickle.dump(samples, open("./viz/present_compartmental_wont_capture/samples.pkl", "wb") )
    pickle.dump(yhats  , open("./viz/present_compartmental_wont_capture/yhats.pkl", "wb") )

    def demographic_stoch( params, N ):
        i0, max_cases_at_start, repo, percent_sus = params
        
        i0    = max(1,i0*max_cases_at_start)
        repo  = 5*(repo+1)
        gamma = 1./12
        
        y  = [ (N*percent_sus-i0), i0]
        dt = 0.1
        infections = [i0]
        for t in np.arange(dt,24,dt):
            s,i = y

            R              =  np.where((t % 24 > 7) , repo, 0)
            infection_rate = (R*gamma)*s*i/N

            new_incs = np.random.poisson( np.clip( dt*infection_rate, 0.001, np.inf) )
            new_sus  = s - new_incs
            
            y = [new_sus, i+new_incs]

            infections.append(new_incs)

        infections = np.array(infections)
        infections = infections.reshape(-1,24).sum(0)

        return infections

    max_cases_at_start = 5#np.max(y[:,0])
    params = zip( samples["I0"], np.repeat(max_cases_at_start, len(samples["I0"])  ), samples["repo"], samples["sus"]  )

    results = Parallel(n_jobs=-1)(delayed(demographic_stoch)(param,8000) for param in params )
    
    all_infections = np.zeros( ( len(results), 24)  )
    for row,result in enumerate(results):
        all_infections[row,:] = result
    pickle.dump(all_infections, open("./viz/present_compartmental_wont_capture/all_infections.pkl", "wb") )
