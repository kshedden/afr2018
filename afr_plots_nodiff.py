"""
Make plots of fitted accuracy curves on probability scale.
"""

import sys
sys.path.insert(
    0, "/afs/umich.edu/user/k/s/kshedden/statsmodels_fork/statsmodels")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
import numpy as np
from afr_data import get_data, get_formula, get_vcf
import patsy
import pickle

# Fit with VB (slow) in addition to Laplace
use_vb = False

# Variance structure
vcs = 1

# Use either adjusted or unadjusted time
adj_time = True

adjs = {True: "adj", False: "noadj"}[adj_time]

vcx = get_vcf(vcs, adj_time)

if use_vb:
    pdf = PdfPages("afr_vb_nodiff_%s.pdf" % adjs)
else:
    pdf = PdfPages("afr_laplace_nodiff_%s.pdf" % adjs)

tname = {True: "bucket_ms_adj_sd", False: "bucket_ms_sd"}[adj_time]

fml = get_formula(adj_time=adj_time)

xl = {True: "Adjusted time (ms)", False: "Time (ms)"}[adj_time]

for group in "oral", "nasal":

    df = get_data(group)

    for outcome in "bucketacc", "bucketcomp":

        yl = {
            "bucketacc": "Target accuracy",
            "bucketcomp": "Competitor accuracy"
        }[outcome]

        fmx = outcome + " ~ " + fml

        model = BinomialBayesMixedGLM.from_formula(
            fmx, vcx, df, vcp_p=3, fe_p=3)

        fid = open(group + ".pkl", "rb")
        pars = pickle.load(fid)
        fid.close()
        if adj_time:
            tm = pars["tm"]
            ts = pars["ts"]
        else:
            tm = pars["tm_adj"]
            ts = pars["ts_adj"]

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
        plt.clf()
        plt.grid(True)
        for participantrace in "kw":
            for speaker in "kw":

                # Save the expanded design matrix for plotting
                dx = df.iloc[0:100, :].copy()
                dx[tname] = np.linspace(dx[tname].min(), dx[tname].max(), 100)
                dx["trial_num_sd"] = 0
                dx["PC1_all_sd"] = 0
                dx["kfirst"] = 0

                kw = int(speaker == "k" and participantrace == "w")
                wk = int(speaker == "w" and participantrace == "k")
                kk = int(speaker == "k" and participantrace == "k")
                ww = int(speaker == "w" and participantrace == "w")

                dx["kw_d_kk"] = kw - kk
                dx["kw_d_ww"] = kw - ww
                dx["wk_d_kk"] = wk - kk

                b = patsy.dmatrix(
                    model.data.design_info, dx, return_type='dataframe')

                q = b.shape[1]
                cov = cov.iloc[0:q, 0:q]
                d = params["mean"].iloc[0:q]
                fv = np.dot(b, d)
                se = []
                for (k, u) in b.iterrows():
                    se.append(np.dot(u, np.dot(cov, u)))
                se = np.sqrt(np.asarray(se))
                fp = 1 / (1 + np.exp(-fv))
                fpl = 1 / (1 + np.exp(-(fv - 2 * se)))
                fpu = 1 / (1 + np.exp(-(fv + 2 * se)))
                label = "p=%s/s=%s" % (participantrace, speaker)
                xx = tm + ts * dx[tname]
                plt.plot(xx, fp, label=label)
                plt.fill_between(xx, fpl, fpu, color='grey', alpha=0.5)

        ha, lb = plt.gca().get_legend_handles_labels()
        leg = plt.figlegend(ha, lb, "upper center", ncol=4)
        leg.draw_frame(False)
        plt.title(group + ": " +
                  ["word intercepts", "word intercepts+slopes"][vcs])
        plt.xlabel(xl, size=16)
        plt.ylabel(yl, size=16)
        plt.ylim(0, 1)
        pdf.savefig()

        # Accuracy against time, by trial number, for speaker/participant
        # groups
        for participantrace in "kw":
            for speaker in "kw":

                kw = int(speaker == "k" and participantrace == "w")
                wk = int(speaker == "w" and participantrace == "k")
                kk = int(speaker == "k" and participantrace == "k")
                ww = int(speaker == "w" and participantrace == "w")

                plt.clf()
                plt.grid(True)
                for tnum in -1, 1:

                    dx["trial_num_sd"] = tnum
                    dx["PC1_all_sd"] = 0

                    dx["kw_d_kk"] = kw - kk
                    dx["kw_d_ww"] = kw - ww
                    dx["wk_d_kk"] = wk - kk

                    b = patsy.dmatrix(
                        model.data.design_info, dx, return_type='dataframe')
                    q = b.shape[1]
                    cov = cov.iloc[0:q, 0:q]
                    d = params["mean"].iloc[0:q]
                    fv = np.dot(b, d)
                    se = []
                    for (k, u) in b.iterrows():
                        se.append(np.dot(u, np.dot(cov, u)))
                    se = np.sqrt(np.asarray(se))
                    fp = 1 / (1 + np.exp(-fv))
                    fpl = 1 / (1 + np.exp(-(fv - 2 * se)))
                    fpu = 1 / (1 + np.exp(-(fv + 2 * se)))

                    label = "Trial number=%.1f" % tnum
                    xx = tm + ts * dx[tname]
                    plt.plot(xx, fp, label=label)
                    plt.fill_between(xx, fpl, fpu, color='grey', alpha=0.5)

                ha, lb = plt.gca().get_legend_handles_labels()
                leg = plt.figlegend(ha, lb, "upper center", ncol=4)
                leg.draw_frame(False)
                plt.title("%s: speaker=%s participant=%s; %s" %
                          (group, speaker, participantrace,
                           ["word intercepts", "word intercepts+slopes"][vcs]))
                plt.xlabel(xl, size=16)
                plt.ylabel(yl, size=16)
                plt.ylim(0, 1)
                pdf.savefig()

        # Accuracy against time at various PC loadings
        for pcd in 25, 40:
            for participantrace in "kw":
                for speaker in "kw":

                    kw = int(speaker == "k" and participantrace == "w")
                    wk = int(speaker == "w" and participantrace == "k")
                    kk = int(speaker == "k" and participantrace == "k")
                    ww = int(speaker == "w" and participantrace == "w")

                    plt.clf()
                    plt.axes([0.1, 0.1, 0.76, 0.8])
                    plt.grid(True)

                    dx = df.iloc[0:100, :].copy()
                    dx[tname] = np.linspace(dx[tname].min(), dx[tname].max(), 100)
                    dx["trial_num_sd"] = 0
                    dx["kfirst"] = 0

                    dx["kw_d_kk"] = kw - kk
                    dx["kw_d_ww"] = kw - ww
                    dx["wk_d_kk"] = wk - kk

                    for f in -1, 1:
                        dx["PC1_all_sd"] = np.percentile(df.PC1_all_sd, 50 + f*pcd)
                        b = patsy.dmatrix(
                            model.data.design_info, dx, return_type='dataframe')
                        q = b.shape[1]
                        cov = cov.iloc[0:q, 0:q]
                        d = params["mean"].iloc[0:q]
                        fv = np.dot(b, d)
                        se = []
                        for (k, u) in b.iterrows():
                            se.append(np.dot(u, np.dot(cov, u)))
                        se = np.sqrt(np.asarray(se))
                        fp = 1 / (1 + np.exp(-fv))
                        fpl = 1 / (1 + np.exp(-(fv - 2 * se)))
                        fpu = 1 / (1 + np.exp(-(fv + 2 * se)))

                        label = "%d" % (50 + f*pcd)
                        xx = tm + ts * dx[tname]
                        plt.plot(xx, fp, label=label)
                        plt.fill_between(xx, fpl, fpu, color='grey', alpha=0.5)

                    ha, lb = plt.gca().get_legend_handles_labels()
                    leg = plt.figlegend(ha, lb, "center right", ncol=1, handletextpad=0.001)
                    leg.draw_frame(False)
                    leg.set_title("PC score")
                    plt.title("%s: speaker=%s participant=%s, %s" %
                              (group, speaker, participantrace,
                               ["word intercepts", "word intercepts+slopes"][vcs]))
                    plt.xlabel(xl, size=16)
                    plt.ylabel(yl, size=16)
                    plt.ylim(0, 1)
                    pdf.savefig()

        # Accuracy against time for kfirst = 0, 1
        for participantrace in "kw":
            for speaker in "kw":

                kw = int(speaker == "k" and participantrace == "w")
                wk = int(speaker == "w" and participantrace == "k")
                kk = int(speaker == "k" and participantrace == "k")
                ww = int(speaker == "w" and participantrace == "w")

                plt.clf()
                plt.grid(True)

                dx = df.iloc[0:100, :].copy()
                dx[tname] = np.linspace(dx[tname].min(), dx[tname].max(), 100)
                dx["trial_num_sd"] = 0
                dx["PC1_all_sd"] = 0

                dx["kw_d_kk"] = kw - kk
                dx["kw_d_ww"] = kw - ww
                dx["wk_d_kk"] = wk - kk

                for kfirst in 0, 1:
                    dx["kfirst"] = kfirst
                    b = patsy.dmatrix(
                        model.data.design_info, dx, return_type='dataframe')
                    q = b.shape[1]
                    cov = cov.iloc[0:q, 0:q]
                    d = params["mean"].iloc[0:q]
                    fv = np.dot(b, d)
                    se = []
                    for (k, u) in b.iterrows():
                        se.append(np.dot(u, np.dot(cov, u)))
                    se = np.sqrt(np.asarray(se))
                    fp = 1 / (1 + np.exp(-fv))
                    fpl = 1 / (1 + np.exp(-(fv - 2 * se)))
                    fpu = 1 / (1 + np.exp(-(fv + 2 * se)))

                    label = "kfirst=%d" % kfirst
                    xx = tm + ts * dx[tname]
                    plt.plot(xx, fp, label=label)
                    plt.fill_between(xx, fpl, fpu, color='grey', alpha=0.5)

                ha, lb = plt.gca().get_legend_handles_labels()
                leg = plt.figlegend(ha, lb, "upper center", ncol=4)
                leg.draw_frame(False)
                plt.title("%s: speaker=%s participant=%s, %s" %
                          (group, speaker, participantrace,
                           ["word intercepts", "word intercepts+slopes"][vcs]))
                plt.xlabel(xl, size=16)
                plt.ylabel(yl, size=16)
                plt.ylim(0, 1)
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
        if w.vname.startswith("C(word)[") and tname in w.vname:
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
