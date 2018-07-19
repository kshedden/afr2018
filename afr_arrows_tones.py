import sys
sys.path.insert(0, "/afs/umich.edu/user/k/s/kshedden/statsmodels_fork/statsmodels")

import numpy as np
import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
import statsmodels.api as sm
import patsy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


vcx = {"participant": "0 + C(participant)"}

out = open("afr_arrows_tones.txt", "w")

pdf = PdfPages("afr_arrows_tones.pdf")

for dt in "tones", "arrows":

    if dt == "arrows":
        fml = "bucketacc ~ (bucket_ms_sd + bs(bucket_ms_sd, df=4)*PC1_all_sd)*participantrace"
        df = pd.read_csv("afrikaans_arrows.csv.gz")
        df = df.loc[df["bucket.ms"] <= 1000, :]
    else:
        fml = "bucketacc ~ (bucket_ms_sd + bs(bucket_ms_sd, df=4)*condition*PC1_all_sd)*participantrace"
        df = pd.read_csv("afrikaans_tones.csv.gz")
        df = df.loc[df["bucket.ms"] <= 2000, :]

    df.columns = [x.replace(".", "_") for x in df.columns]

    tm = df.bucket_ms.mean()
    ts = df.bucket_ms.std()
    df["bucket_ms_sd"] = (df.bucket_ms - tm) / ts
    df["PC1_all_sd"] = (df.PC1_all - df.PC1_all.mean()) / df.PC1_all.std()

    model1 = BinomialBayesMixedGLM.from_formula(fml, vcx, df, vcp_p=3, fe_p=3)

    # Posterior mode fit using Laplace approximation
    result1 = model1.fit_map()
    out.write("%s:\n" % dt)
    out.write(result1.summary().as_text())
    mean = np.concatenate((result1.fe_mean, result1.vcp_mean, result1.vc_mean))
    sd = np.concatenate((result1.fe_sd, result1.vcp_sd, result1.vc_sd))
    pn = model1.fep_names + model1.vcp_names + model1.vc_names
    rx = pd.DataFrame({"mean": mean, "sd": sd}, index=pn)
    rx.to_csv("%s_params_map.csv" % dt)
    cov = result1.cov_params()

    if dt == "arrows":

        for participantrace in "wk":

            dxa = []
            for pcs in -1, 1:
                dx = df.iloc[0:100, :].copy()
                dx["bucket_ms_sd"] = np.linspace(dx.bucket_ms_sd.min(), dx.bucket_ms_sd.max(), 100)
                dx["participantrace"] = participantrace
                dx["PC1_all_sd"] = pcs
                dxa.append(dx)

            c = []
            for dd in dxa:
                c.append(patsy.dmatrix(model1.data.design_info, dd, return_type='dataframe'))

            plt.clf()
            plt.axes([0.1, 0.1, 0.78, 0.8])
            plt.grid(True)
            for j,b in enumerate(c):

                q = b.shape[1]
                cov = cov.iloc[0:q, 0:q]
                d = result1.params[0:q]
                fv = np.dot(b, d)
                se = []
                for (k, u) in b.iterrows():
                    se.append(np.dot(u, np.dot(cov, u)))
                se = np.sqrt(np.asarray(se))
                fvl = fv - 2*se
                fvu = fv + 2*se

                fv = 1 / (1 + np.exp(-fv))
                fvl = 1 / (1 + np.exp(-fvl))
                fvu = 1 / (1 + np.exp(-fvu))

                xx = tm + ts*dx.bucket_ms_sd
                plt.plot(xx, fv, label=str(2*j-1))
                plt.fill_between(xx, fvl, fvu, color='grey', alpha=0.5)

            ha, lb = plt.gca().get_legend_handles_labels()
            leg = plt.figlegend(ha, lb, "center right")
            leg.set_title("PC")
            leg.draw_frame(False)
            plt.xlabel("Time (ms)", size=16)
            plt.title("PC's for %s, participantrace=%s\n" % (dt, participantrace))
            plt.ylabel("Accuracy", size=16)

            pdf.savefig()

            b = c[1] - c[0]
            plt.clf()
            plt.axes([0.1, 0.1, 0.78, 0.8])
            plt.grid(True)
            q = b.shape[1]
            cov = cov.iloc[0:q, 0:q]
            d = result1.params[0:q]
            fv = np.dot(b, d)
            se = []
            for (k, u) in b.iterrows():
                se.append(np.dot(u, np.dot(cov, u)))
            se = np.sqrt(np.asarray(se))
            fvl = fv - 2*se
            fvu = fv + 2*se

            xx = tm + ts*dx.bucket_ms_sd
            plt.plot(xx, fv)
            plt.fill_between(xx, fvl, fvu, color='grey', alpha=0.5)

            plt.xlabel("Time (ms)", size=16)
            plt.title("PC contrast for %s, participantrace=%s\n" % (dt, participantrace))
            plt.ylabel("Logit accuracy difference", size=16)

            pdf.savefig()

    else:

        for participantrace in "kw":
            for cond in df.condition.unique():

                dxa = []
                for pcs in -1, 1:
                    dx = df.iloc[0:100, :].copy()
                    dx["bucket_ms_sd"] = np.linspace(dx.bucket_ms_sd.min(), dx.bucket_ms_sd.max(), 100)
                    dx["PC1_all_sd"] = pcs
                    dx["participantrace"] = participantrace
                    dx["condition"] = cond
                    dxa.append(dx)

                c = []
                for dd in dxa:
                    c.append(patsy.dmatrix(model1.data.design_info, dd, return_type='dataframe'))

                plt.clf()
                plt.axes([0.1, 0.1, 0.78, 0.8])
                plt.grid(True)
                for j, b in enumerate(c):
                    q = b.shape[1]
                    cov = cov.iloc[0:q, 0:q]
                    d = result1.params[0:q]
                    fv = np.dot(b, d)
                    se = []
                    for (k, u) in b.iterrows():
                        se.append(np.dot(u, np.dot(cov, u)))
                    se = np.sqrt(np.asarray(se))
                    fvl = fv - 2*se
                    fvu = fv + 2*se

                    fv = 1 / (1 + np.exp(-fv))
                    fvl = 1 / (1 + np.exp(-fvl))
                    fvu = 1 / (1 + np.exp(-fvu))

                    xx = tm + ts*dx.bucket_ms_sd
                    plt.plot(xx, fv, label=str(2*j-1))
                    plt.fill_between(xx, fvl, fvu, color='grey', alpha=0.5)

                plt.xlabel("Time (ms)", size=16)

                ha, lb = plt.gca().get_legend_handles_labels()
                leg = plt.figlegend(ha, lb, "center right")
                leg .set_title("PC")
                leg.draw_frame(False)

                plt.title("PC's for %s, condition=%s, participantrace=%s\n" % (dt, cond, participantrace))
                plt.ylabel("Accuracy", size=16)

                pdf.savefig()

                b = c[1] - c[0]
                plt.clf()
                plt.axes([0.1, 0.1, 0.78, 0.8])
                plt.grid(True)
                q = b.shape[1]
                cov = cov.iloc[0:q, 0:q]
                d = result1.params[0:q]
                fv = np.dot(b, d)
                se = []
                for (k, u) in b.iterrows():
                    se.append(np.dot(u, np.dot(cov, u)))
                se = np.sqrt(np.asarray(se))
                fvl = fv - 2*se
                fvu = fv + 2*se

                xx = tm + ts*dx.bucket_ms_sd
                plt.plot(xx, fv)
                plt.fill_between(xx, fvl, fvu, color='grey', alpha=0.5)

                plt.xlabel("Time (ms)", size=16)

                plt.title("PC contrast for %s, condition=%s, participantrace=%s\n" % (dt, cond, participantrace))
                plt.ylabel("Logit accuracy difference", size=16)

                pdf.savefig()


out.close()
pdf.close()