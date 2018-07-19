import sys
sys.path.insert(
    0, "/afs/umich.edu/user/k/s/kshedden/statsmodels_fork/statsmodels")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
import numpy as np
from afr_data import get_data, get_formula, vcf
import patsy
import pickle

# Use VB versus Laplace approximation for fitting
use_vb = False

# The variance component structure
vcs = 1

# Use adjusted or unadjusted time
adj_time = False

adjs = {True: "adj", False: "noadj"}[adj_time]

if use_vb:
    pdf = PdfPages("afr_vb_%s.pdf" % adjs)
else:
    pdf = PdfPages("afr_laplace_%s.pdf" % adjs)

tname = {True: "bucket_ms_adj_sd", False: "bucket_ms_sd"}[adj_time]

for group in "oral", "nasal":

    df = get_data(group)
    for outcome in "bucketacc", "bucketcomp":

        fml = get_formula(adj_time=adj_time)
        fmx = outcome + " ~ " + fml

        yl = {
            "bucketacc": "target accuracy",
            "bucketcomp": "competitor accuracy"
        }[outcome]

        vcx = get_vcf(vcs, adj_time)

        fmx = outcome + " ~ " + fml
        model = BinomialBayesMixedGLM.from_formula(
            fmx, vcx, df, vcp_p=3, fe_p=3)

        fid = open(group + ".pkl", "rb")
        pars = pickle.load(fid)
        fid.close()
        if adj_time:
            tm = pars["tm_adj"]
            ts = pars["ts_adj"]
        else:
            tm = pars["tm"]
            ts = pars["ts"]

        if use_vb:
            params = pd.read_csv(
                "%s_params_%d_%s_%s_vb.csv" % (group, vcs, outcome, adjs))
        else:
            params = pd.read_csv(
                "%s_params_%d_%s_%s_map.csv" % (group, vcs, outcome, adjs))
            cov = pd.read_csv(
                "%s_params_%d_%s_%s_cov.csv" % (group, vcs, outcome, adjs),
                index_col=0)

        col = params.columns.tolist()
        col[0] = "vname"
        params.columns = col

        # Accuracy against time, for speaker/participant race combinations
        for j, x in enumerate([("kk", "kw"), ("ww", "kw"), ("kk", "wk"),
                               ("ww", "wk")]):

            dxa = []
            for k in 0, 1:
                dx = df.iloc[0:100, :].copy()
                dx[tname] = np.linspace(dx[tname].min(), dx[tname].max(), 100)
                dx["trial_num_sd"] = 0
                dx["PC1_all_sd"] = 0

                dx["kw_d_kk"] = int(x[k] == "kw") - int(x[k] == "kk")
                dx["kw_d_ww"] = int(x[k] == "kw") - int(x[k] == "ww")
                dx["wk_d_kk"] = int(x[k] == "wk") - int(x[k] == "kk")

                dxa.append(dx)

            b1 = patsy.dmatrix(
                model.data.design_info, dxa[0], return_type='dataframe')
            b2 = patsy.dmatrix(
                model.data.design_info, dxa[1], return_type='dataframe')
            b = b1 - b2

            q = b.shape[1]
            cov = cov.iloc[0:q, 0:q]

            d = params["mean"].iloc[0:q]
            fv = np.dot(b, d)
            se = []
            for (k, u) in b.iterrows():
                se.append(np.dot(u, np.dot(cov, u)))
            se = np.sqrt(np.asarray(se))
            fvl = fv - 2 * se
            fvu = fv + 2 * se
            label = "Speaker/Participant: %s minus %s" % (x[0], x[1])
            xx = tm + ts * dx[tname]
            plt.clf()
            plt.grid(True)
            plt.plot(xx, fv, label=label)
            plt.fill_between(xx, fvl, fvu, color='grey', alpha=0.5)
            plt.title(label + "\n" + group + ": " +
                      ["word intercepts", "word intercepts+slopes"][vcs])
            plt.xlabel("Time (ms)", size=16)
            plt.ylabel("Logit %s difference" % yl, size=16)
            pdf.savefig()

        #
        # Difference of differences
        #

        li = ("ww", "kw", "kk", "wk")
        for x in li:
            dx = df.iloc[0:100, :].copy()
            dx[tname] = np.linspace(dx[tname].min(), dx[tname].max(), 100)
            dx["trial_num_sd"] = 0
            dx["PC1_all_sd"] = 0

            dx["kw_d_kk"] = int(x == "kw") - int(x == "kk")
            dx["kw_d_ww"] = int(x == "kw") - int(x == "ww")
            dx["wk_d_kk"] = int(x == "wk") - int(x == "kk")

            dxa.append(dx)

        b = []
        for dd in dxa:
            b.append(
                patsy.dmatrix(
                    model.data.design_info, dd, return_type='dataframe'))
        b = b[0] - b[1] - (b[2] - b[3])

        q = b.shape[1]
        cov = cov.iloc[0:q, 0:q]

        d = params["mean"].iloc[0:q]
        fv = np.dot(b, d)
        se = []
        for (k, u) in b.iterrows():
            se.append(np.dot(u, np.dot(cov, u)))
        se = np.sqrt(np.asarray(se))
        fvl = fv - 2 * se
        fvu = fv + 2 * se
        xx = tm + ts * dx[tname]
        plt.clf()
        plt.grid(True)
        plt.plot(xx, fv)
        plt.fill_between(xx, fvl, fvu, color='grey', alpha=0.5)
        plt.title("Speaker/Participant: (kw-ww) - (wk-kk)\n" + group + ": " +
                  ["word intercepts", "word intercepts+slopes"][vcs])
        plt.xlabel("Time (ms)", size=16)
        plt.ylabel("Logit %s difference" % yl, size=16)
        pdf.savefig()

        # Accuracy against time, for single speaker/participant race combinations, differenced
        # by trial (learning rate)
        for j, x in enumerate(["kk", "kw", "ww", "wk"]):

            dxa = []
            for tnum in -1, 1:
                dx = df.iloc[0:100, :].copy()
                dx[tname] = np.linspace(dx[tname].min(), dx[tname].max(), 100)
                dx["trial_num_sd"] = tnum
                dx["PC1_all_sd"] = 0

                dx["kw_d_kk"] = int(x == "kw") - int(x == "kk")
                dx["kw_d_ww"] = int(x == "kw") - int(x == "ww")
                dx["wk_d_kk"] = int(x == "wk") - int(x == "kk")

                dxa.append(dx)

            c = []
            for dd in dxa:
                c.append(
                    patsy.dmatrix(
                        model.data.design_info, dd, return_type='dataframe'))
            b = c[1] - c[0]

            plt.clf()
            plt.axes([0.1, 0.1, 0.78, 0.8])
            plt.grid(True)
            q = b.shape[1]
            cov = cov.iloc[0:q, 0:q]
            d = params["mean"].iloc[0:q]
            fv = np.dot(b, d)
            se = []
            for (k, u) in b.iterrows():
                se.append(np.dot(u, np.dot(cov, u)))
            se = np.sqrt(np.asarray(se))
            fvl = fv - 2 * se
            fvu = fv + 2 * se

            xx = tm + ts * dx[tname]
            plt.plot(xx, fv)
            plt.fill_between(xx, fvl, fvu, color='grey', alpha=0.5)

            plt.title(
                "Speaker/Participant: %s trial contrast\n%s: %s" %
                (x, group, ["word intercepts", "word intercepts+slopes"][vcs]))
            plt.xlabel("Time (ms)", size=16)
            plt.ylabel("Logit %s difference" % yl, size=16)
            pdf.savefig()

        # Accuracy against time, by trial number, for speaker/participant
        # groups
        # Accuracy against time, for speaker/participant race combinations
        for j, x in enumerate([("kk", "kw"), ("ww", "kw"), ("kk", "wk"),
                               ("ww", "wk")]):

            dxa = []
            for tnum in -1, 1:
                for k in 0, 1:
                    dx = df.iloc[0:100, :].copy()
                    dx[tname] = np.linspace(dx[tname].min(), dx[tname].max(),
                                            100)
                    dx["trial_num_sd"] = tnum
                    dx["PC1_all_sd"] = 0

                    dx["kw_d_kk"] = int(x[k] == "kw") - int(x[k] == "kk")
                    dx["kw_d_ww"] = int(x[k] == "kw") - int(x[k] == "ww")
                    dx["wk_d_kk"] = int(x[k] == "wk") - int(x[k] == "kk")

                    dxa.append(dx)

            c = []
            for dd in dxa:
                c.append(
                    patsy.dmatrix(
                        model.data.design_info, dd, return_type='dataframe'))
            b = c[2] - c[3] - (c[0] - c[1])

            plt.clf()
            plt.axes([0.1, 0.1, 0.78, 0.8])
            plt.grid(True)
            q = b.shape[1]
            cov = cov.iloc[0:q, 0:q]
            d = params["mean"].iloc[0:q]
            fv = np.dot(b, d)
            se = []
            for (k, u) in b.iterrows():
                se.append(np.dot(u, np.dot(cov, u)))
            se = np.sqrt(np.asarray(se))
            fvl = fv - 2 * se
            fvu = fv + 2 * se

            xx = tm + ts * dx[tname]
            plt.plot(xx, fv)
            plt.fill_between(xx, fvl, fvu, color='grey', alpha=0.5)

            plt.title("Speaker/Participant: %s minus %s trial contrast\n%s: %s"
                      % (x[0], x[1], group,
                         ["word intercepts", "word intercepts+slopes"][vcs]))
            plt.xlabel("Time (ms)", size=16)
            plt.ylabel("Logit %s difference" % yl, size=16)
            pdf.savefig()

        # Accuracy against time, by PC score, for speaker/participant groups
        for j, x in enumerate(["kk", "kw", "ww", "wk"]):

            dxa = []
            for pcs in -1, 1:
                dx = df.iloc[0:100, :].copy()
                dx[tname] = np.linspace(dx[tname].min(), dx[tname].max(), 100)
                dx["trial_num_sd"] = 0
                dx["PC1_all_sd"] = pcs

                dx["kw_d_kk"] = int(x == "kw") - int(x == "kk")
                dx["kw_d_ww"] = int(x == "kw") - int(x == "ww")
                dx["wk_d_kk"] = int(x == "wk") - int(x == "kk")

                dxa.append(dx)

            c = []
            for dd in dxa:
                c.append(
                    patsy.dmatrix(
                        model.data.design_info, dd, return_type='dataframe'))
            b = c[1] - c[0]

            plt.clf()
            plt.axes([0.1, 0.1, 0.78, 0.8])
            plt.grid(True)
            q = b.shape[1]
            cov = cov.iloc[0:q, 0:q]
            d = params["mean"].iloc[0:q]
            fv = np.dot(b, d)
            se = []
            for (k, u) in b.iterrows():
                se.append(np.dot(u, np.dot(cov, u)))
            se = np.sqrt(np.asarray(se))
            fvl = fv - 2 * se
            fvu = fv + 2 * se

            xx = tm + ts * dx[tname]
            plt.plot(xx, fv)
            plt.fill_between(xx, fvl, fvu, color='grey', alpha=0.5)

            plt.title(
                "Speaker/Participant: %s PC contrast\n%s: %s" %
                (x, group, ["word intercepts", "word intercepts+slopes"][vcs]))
            plt.xlabel("Time (ms)", size=16)
            plt.ylabel("Logit %s difference" % yl, size=16)
            pdf.savefig()

        # Accuracy against time, by PC score, for speaker/participant groups
        for j, x in enumerate([("kk", "kw"), ("ww", "kw"), ("kk", "wk"),
                               ("ww", "wk")]):

            dxa = []
            for pcs in -1, 1:
                for k in 0, 1:
                    dx = df.iloc[0:100, :].copy()
                    dx[tname] = np.linspace(dx[tname].min(), dx[tname].max(),
                                            100)
                    dx["trial_num_sd"] = 0
                    dx["PC1_all_sd"] = pcs

                    dx["kw_d_kk"] = int(x[k] == "kw") - int(x[k] == "kk")
                    dx["kw_d_ww"] = int(x[k] == "kw") - int(x[k] == "ww")
                    dx["wk_d_kk"] = int(x[k] == "wk") - int(x[k] == "kk")

                    dxa.append(dx)

            c = []
            for dd in dxa:
                c.append(
                    patsy.dmatrix(
                        model.data.design_info, dd, return_type='dataframe'))
            b = c[2] - c[3] - (c[0] - c[1])

            plt.clf()
            plt.axes([0.1, 0.1, 0.78, 0.8])
            plt.grid(True)
            q = b.shape[1]
            cov = cov.iloc[0:q, 0:q]
            d = params["mean"].iloc[0:q]
            fv = np.dot(b, d)
            se = []
            for (k, u) in b.iterrows():
                se.append(np.dot(u, np.dot(cov, u)))
            se = np.sqrt(np.asarray(se))
            fvl = fv - 2 * se
            fvu = fv + 2 * se

            xx = tm + ts * dx[tname]
            plt.plot(xx, fv)
            plt.fill_between(xx, fvl, fvu, color='grey', alpha=0.5)

            plt.title("Speaker/Participant: %s minus %s PC contrast\n%s: %s" %
                      (x[0], x[1], group,
                       ["word intercepts", "word intercepts+slopes"][vcs]))
            plt.xlabel("Time (ms)", size=16)
            plt.ylabel("Logit %s difference" % yl, size=16)
            pdf.savefig()

    # Histogram of participant effects
    pe = []
    for i in range(params.shape[0]):
        p = params.iloc[i, :]
        if p.vname.startswith("C(participant)["):
            pe.append(p["mean"])
    pe = np.asarray(pe)
    plt.clf()
    plt.hist(pe)
    plt.title("%s participant effects" % group)
    plt.xlabel("Posterior mean", size=15)
    pdf.savefig()

    # Histogram of word effects
    we = []
    for i in range(params.shape[0]):
        w = params.iloc[i, :]
        if w.vname.startswith("C(word)[") and ":" not in w.vname:
            we.append(w["mean"])
    we = np.asarray(we)
    plt.clf()
    plt.hist(we)
    plt.title("%s word effects" % group)
    plt.xlabel("Posterior mean", size=15)
    pdf.savefig()

    # Histogram of word slope effects
    ws = []
    for i in range(params.shape[0]):
        w = params.iloc[i, :]
        if w.vname.startswith("C(word)[") and "bucket_ms_adj" in w.vname:
            ws.append(w["mean"])
    if len(ws) > 0:
        ws = np.asarray(ws)
        plt.clf()
        plt.hist(ws)
        plt.title("%s word slopes" % group)
        plt.xlabel("Posterior mean", size=15)
        pdf.savefig()

        plt.clf()
        plt.grid(True)
        plt.plot(we, ws, 'o')
        plt.xlabel("Word intercept", size=15)
        plt.ylabel("Word slope", size=15)
        pdf.savefig()

pdf.close()
