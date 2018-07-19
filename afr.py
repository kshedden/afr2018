import sys
sys.path.insert(0, "/afs/umich.edu/user/k/s/kshedden/statsmodels_fork/statsmodels")

import numpy as np
import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from afr_data import get_data, get_formula, get_vcf
import statsmodels.api as sm

# Use VB for fitting, in addition to Laplace.
use_vb = False

# Variance component structure
vcs = 1

# Use either adjusted or unadjusted time
adj_time = False

adjs = {True: "adj", False: "noadj"}[adj_time]
out = open("afr_results_%s.txt" % adjs, "w")

for group in "oral", "nasal":
    for outcome in "bucketacc", "bucketcomp":

        vcx = get_vcf(vcs, adj_time)

        df = get_data(group)

        fml = get_formula(adj_time=adj_time)
        fmx = outcome + " ~ " + fml

        model1 = BinomialBayesMixedGLM.from_formula(fmx, vcx, df, vcp_p=3, fe_p=3)

        if use_vb:
            model2 = BinomialBayesMixedGLM.from_formula(fmx, vcx, df, vcp_p=3, fe_p=3)

        # Posterior mode fit using Laplace approximation
        result1 = model1.fit_map()
        out.write("group=%s vcs=%d outcome=%s adj=%s:\n" % (group, vcs, outcome, adjs))
        out.write(result1.summary().as_text() + "\n\n")
        mean = np.concatenate((result1.fe_mean, result1.vcp_mean, result1.vc_mean))
        sd = np.concatenate((result1.fe_sd, result1.vcp_sd, result1.vc_sd))
        pn = model1.fep_names + model1.vcp_names + model1.vc_names
        rx = pd.DataFrame({"mean": mean, "sd": sd}, index=pn)
        rx.to_csv("%s_params_%d_%s_%s_map.csv" % (group, vcs, outcome, adjs))
        cm = result1.cov_params()
        cm.to_csv("%s_params_%d_%s_%s_cov.csv" % (group, vcs, outcome, adjs))

        # VB fit, takes about two hours
        if use_vb:
            result2 = model2.fit_vb(result1.params, np.ones_like(result1.params), verbose=True)
            out.write("\n\n")
            out.write(result2.summary().as_text())
            out.write("\n\n")
            mean = np.concatenate((result2.fe_mean, result2.vcp_mean, result2.vc_mean))
            sd = np.concatenate((result2.fe_sd, result2.vcp_sd, result2.vc_sd))
            pn = model2.fep_names + model2.vcp_names + model2.vc_names
            rx = pd.DataFrame({"mean": mean, "sd": sd}, index=pn)
            rx.to_csv("%s_params_%d_%s_%s_vb.csv" % (group, vcs, outcome, adjs))

out.close()
