import numpy as np
import pandas as pd
import pickle

# Fixed effects formula
fml_base = "bucket_ms_adj_sd + bs(bucket_ms_adj_sd, df=4)*(kw_d_kk + kw_d_ww + wk_d_kk)*trial_num_sd + "
fml_base += "bs(bucket_ms_adj_sd, df=4)*bs(PC1_all_sd, df=4) + kfirst*bs(bucket_ms_adj_sd, df=4)"


def get_formula(adj_time=True):

    fml = fml_base

    if not adj_time:
        return fml_base.replace("bucket_ms_adj", "bucket_ms")

    return fml


fname = "afrikaans_perc_data.csv.gz"

def get_vcf(vcs, adj_time):

    vcf = {
        "participant": "0 + C(participant)",
        "word": "0 + C(word)",
        "wordslope": "0 + C(word):bucket_ms_adj_sd"
    }

    if vcs == 0:
        del vcf["wordslope"]

    if not adj_time:
        for k, v in vcf.items():
            vcf[k] = vcf[k].replace("bucket_ms_adj_sd", "bucket_ms_sd")

    return vcf


# group is "oral" or "nasal"
def get_data(group):

    df = pd.read_csv(fname)
    df = df.loc[df.type == group, :]

    # Can't have . in pandas names
    df.columns = [x.replace(".", "_") for x in df.columns]

    # Create ethnic group contrasts
    df["speakerpart"] = df.speaker + df.participantrace

    df["kw_d_kk"] = 0
    df.loc[df.speakerpart == "kw", "kw_d_kk"] = 1
    df.loc[df.speakerpart == "kk", "kw_d_kk"] = -1

    df["kw_d_ww"] = 0
    df.loc[df.speakerpart == "kw", "kw_d_ww"] = 1
    df.loc[df.speakerpart == "ww", "kw_d_ww"] = -1

    df["wk_d_kk"] = 0
    df.loc[df.speakerpart == "wk", "wk_d_kk"] = 1
    df.loc[df.speakerpart == "kk", "wk_d_kk"] = -1

    df["wk_d_ww"] = 0
    df.loc[df.speakerpart == "wk", "wk_d_ww"] = 1
    df.loc[df.speakerpart == "ww", "wk_d_ww"] = -1

    # Standardize covariates
    df["PC1_all_sd"] = (df.PC1_all - df.PC1_all.mean()) / df.PC1_all.std()
    df["trial_num_sd"] = (
        df.trial_num - df.trial_num.mean()) / df.trial_num.std()
    tm_adj = df.bucket_ms_adj.mean()
    ts_adj = df.bucket_ms_adj.std()
    df["bucket_ms_adj_sd"] = (df.bucket_ms_adj - tm_adj) / ts_adj

    tm = df.bucket_ms.mean()
    ts = df.bucket_ms.std()
    df["bucket_ms_sd"] = (df.bucket_ms - tm) / ts

    fid = open(group + ".pkl", "wb")
    pickle.dump({"tm_adj": tm_adj, "ts_adj": ts_adj, "tm": tm, "ts": ts}, fid)
    fid.close()

    return df
