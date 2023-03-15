import argparse
import os
import random
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader

from comic.data_utils import MultiOmicsDataset, OmicsDataset
from comic.models import MultiPIMKLNet, PIMKLNet

# path macros (need to be updated if run on a different system)
BASELINE = "/home/jonas/Documents/Research/Comik/Data/PIMKL/Cunetal/aucs_Cunetal.csv"
PATHWAYS = (
    "/home/jonas/Documents/Research/Comik/Data/PIMKL/Cunetal/{}/KEGG_PC/h.all/inducers/"
)
PIMKL_CUNETAL = (
    "/home/jonas/Documents/Research/Comik/Data/PIMKL/Cunetal/{}/KEGG_PC/h.all/"
    + "auc_{}_gene_expression_kegg_pc_relapse<5years_hallmark_EasyMKL_splits=Cunetal.csv"
)
PIMKL_METABRIC = (
    "/home/jonas/Documents/Research/Comik/Data/PIMKL/METABRIC/"
    + "auc_{}_{}_dfs_status_hallmark_cv=100_mc=20_EasyMKL.csv"
)
RESULT = "/home/jonas/Documents/Research/Comik/cluster_results/{}/{}/"
DUMMY = "/home/jonas/Documents/Research/Comik/Data/benchmark_dataset_size/"


def load_args():
    """Command line parser for analysis script.
    """
    parser = argparse.ArgumentParser(description="Comik Analysis Script")
    parser.add_argument(
        "--type",
        dest="type",
        default="cunetal",
        type=str,
        choices=[
            "cunetal",
            "metabric",
            "bic",
            "time",
            "multitime",
            "gridsearch",
            "preprocessing",
            "global",
            "local",
        ],
        help="Select the type of analysis that will be performed.",
    )
    parser.add_argument(
        "--run",
        dest="run",
        type=str,
        help="Select the experiment that will be analyzed.",
    )
    parser.add_argument(
        "--network",
        dest="network",
        default="pimkl",
        type=str,
        choices=["pimkl", "adapimkl", "glpimkl", "laplacian", "multipimkl"],
        help="Select the typ of network architecture to be used for the experiment.",
    )
    parser.add_argument(
        "--data",
        dest="data",
        type=str,
        nargs="+",
        help=(
            "Paths to the files containing data and labels."
            "The last provided path will be treated as the labels path."
            " This argument is only used to compute local interpretations "
            "(i.e. if --type is set to local)."
        ),
    )

    # parse arguments
    args = parser.parse_args()

    # update network argument
    if args.network == "pimkl":
        args.network = ("PIMKLNet", "pimkl")
    elif args.network == "adapimkl":
        args.network = ("AdaPIMKLNet", "adapimkl")
    elif args.network == "glpimkl":
        args.network = ("GLPIMKLNet", "glpimkl")
    elif args.network == "laplacian":
        args.network = ("GLKNet", "laplacian")
    elif args.network == "multipimkl":
        args.network = ("MultiPIMKLNet", "multipimkl")

    return args


def _count_metabric_labels():
    """This function is a simple helper to count the samples per class
    for the Metabric cohort. For this to work, the labels of the Metabric
    cohort have to be prepared using the appropriate function calls.
    """
    s_file = (
        "/home/jonas/Documents/Research/Comik/Data/PIMKL/METABRIC/cna/cna_kegg-pc.csv"
    )
    l_file = "/home/jonas/Documents/Research/Comik/Data/PIMKL/METABRIC/cna/cna_kegg-pc_labels.csv"

    # get all the available samples
    samples = []
    with open(s_file, "r") as in_file:
        for i, line in enumerate(in_file):
            # skip the first line
            if i == 0:
                continue

            # sample id is the first entry in line
            samples.append(line.split(",")[0])

    # count the number of samples per class
    pos = 0
    neg = 0
    with open(l_file, "r") as in_file:
        for line in in_file:
            sample = line.split(",")[0]
            label = line.split(",")[1].strip()

            if sample in samples:
                if label == "1:Recurred":
                    pos += 1
                else:
                    neg += 1

    print(f"Samples in Metabric Cohort: {len(samples)}")
    print(f"    Samples in Recurred Class: {pos}")
    print(f"    Samples in Non-Recurred Class: {neg}")


def _count_bic_labels():
    """This function is a simple helper to count the positive and negative
    class for the BIC multi-omics dataset. This function assumes that the 
    files are already preprocessed. If you want to use this function, please
    make the appropriate changes to the main function."""
    available_sample_file = (
        "/home/jonas/Documents/Research/Comik/Data/breast/exp_preprocessed.csv"
    )
    label_file = "/home/jonas/Documents/Research/Comik/Data/breast/labels.csv"

    # read in the first line of the exp file
    with open(available_sample_file, "r") as in_file:
        samples = in_file.readline().split(",")
        samples = [s.strip('"') for s in samples]

    # count the classes
    neg = 0
    pos = 0
    with open(label_file, "r") as l_file:
        for line in l_file:
            id, label = line.split(",")
            if id in samples:
                if label.strip() == "True":
                    pos += 1
                else:
                    neg += 1

    print(f"{pos + neg} Patients in BIC dataset")
    print(f"    {pos} Patients have a dmfs < 5 years")
    print(f"    {neg} Patients have a dmfs >= 5 years")


def _prepare_label_file(
    filepath_in,
    filepath_out,
    target_column="days_to_new_tumor_event_after_initial_treatment",
    sep="\t",
):
    """Helper function to prepare the labels file of the multiomics datasets.
    """
    with open(filepath_in, "r") as in_file:
        with open(filepath_out, "w") as out_file:

            # read in the input file line by line
            for i, line in enumerate(in_file):

                # convert the line into a list
                linelist = line.strip().split(sep)

                # first line contains the header
                #   -> determine the index of the column containing the label information
                if i == 0:
                    label_index = linelist.index(target_column)
                    continue

                # get the id of the current sample
                sample_id = linelist[0].replace("-", ".")

                # determine the label
                #   -> the binary label is disease-free survival > 5 years, i.e. if the
                #      entry in the target column is either 'NA' or >1820 the label will be True
                try:
                    if int(linelist[label_index]) < 1820:
                        label = "True"
                    else:
                        label = "False"
                except ValueError:
                    if linelist[label_index] == "NA":
                        label = "False"
                    else:
                        label = "True"

                # write id and label to output file
                out_file.write(f"{sample_id},{label}\n")


def _prepare_data_file(filepath_in, filepath_out, gene_index, sample_index, sep=","):
    """Helper function to reorder the samples and genes in a data file to comply with the order in the
    provided template.
    """
    missing_gene = 0
    found_gene = 0
    found_gene_positions = []
    with open(filepath_in, "r") as in_file:
        with open(filepath_out, "w") as out_file:

            # create list to temporarily store the lines of the output file
            outlines = [""] * (len(gene_index) + 1)
            for key, value in gene_index.items():
                outlines[value] = f'"{key}",' + ",".join(
                    [str(0.001)] * len(sample_index)
                )

            # iterate over input file
            for i, line in enumerate(in_file):

                # the first line contains the ids
                if i == 0:
                    s_ids = line.strip().split(sep)

                    # concatenate the sample ids in the specified order
                    aux_str = []
                    for sidx in sample_index:
                        aux_str.append(s_ids[sidx])
                    outlines[i] = ",".join(aux_str)

                    continue

                # identify the gene of the current line
                gene = line.split(sep)[0].strip('"').split("|")[0]
                feature_values = line.strip().split(sep)[1:]

                try:
                    # get the index of the current gene, if the gene is available in the template
                    aux_idx = gene_index[gene]

                    # concatenate the feature values in the correct order
                    aux_str = []
                    for sidx in sample_index:
                        aux_str.append(feature_values[sidx])

                    # write the concatenated feature values at the correct position in the
                    # output file
                    outlines[aux_idx] = '"' + gene + '",' + ",".join(aux_str)

                    # store information to be shown to the user
                    found_gene += 1
                    found_gene_positions.append(gene_index[gene])

                except KeyError:
                    # print(f"    {gene} not in gene_index dictionary")
                    missing_gene += 1

            print(f"    {found_gene} genes found in gene_index dictionary")
            print(f"    {missing_gene} genes not in gene_index dictionary")
            print(f"    available gene positions: ", found_gene_positions)

            # write result to out file
            out_file.write("\n".join(outlines))


def prepare_multiomics():
    """This function prepares a multiomics dataset taken from http://acgt.cs.tau.ac.il/multi_omic_benchmark/download.html
    and ensures the format to be compatible with the Laplacian from the PIMKL paper. Please be aware that 
    only the gene expression and DNA methylation data will be prepared and that the DNA methylation data has to be
    group by genes to be applicable. Grouping by genes can be achived with any tool, e.g. RnBeads.
    """
    # paths the each data file and the label file
    #   ATTENTION: These paths have to be adjusted on each computer system
    EXP_FILE = "/home/jonas/Documents/Research/Comik/Data/breast/exp"
    EXP_OUT = EXP_FILE + "_preprocessed.csv"
    METHY_FILE = "/home/jonas/Documents/Research/Comik/Data/breast/methy.csv"
    METHY_OUT = (
        "/home/jonas/Documents/Research/Comik/Data/breast/methy_preprocessed.csv"
    )
    ID_TEMPLATE = (
        "/home/jonas/Documents/Research/Comik/Data/PIMKL/Cunetal/GSE11121/GSE11121.csv"
    )
    LABEL_FILE = "/home/jonas/Documents/Research/Comik/Data/clinical/breast"
    LABEL_OUT = "/home/jonas/Documents/Research/Comik/Data/breast/labels.csv"

    # prepare the label file
    print("Preparing labels...")
    _prepare_label_file(LABEL_FILE, LABEL_OUT)

    # load the index of each gene from the template
    print("\nGetting gene order from the template...")
    gene_index = {}
    with open(ID_TEMPLATE, "r") as id_temp_file:
        for i, line in enumerate(id_temp_file):

            # skip the first line with the sample ids
            if line[1:4] == "GSM":
                continue

            # get the gene symbol of the current feature
            gene_symbol = line.split(",")[0].strip('"')
            gene_index[gene_symbol] = i

    # read in the ids of gene expression and methylation samples
    with open(EXP_FILE, "r") as exp_in_file:
        available_exp = exp_in_file.readline().strip().split(" ")
    with open(METHY_FILE, "r") as methy_in_file:
        available_methy = methy_in_file.readline().strip().split(",")

    # determine which samples have both, gene expression and methylation data
    aux_methy_set = set(available_methy)
    aux_exp_set = set(available_exp)
    common_samples = aux_exp_set.intersection(aux_methy_set)

    # get the indices for the samples that have both data types
    idx_samples_methy = [available_methy.index(i) for i in common_samples]
    idx_samples_exp = [available_exp.index(i) for i in common_samples]

    # open the methylation data, identify all patient with available methylation data, and
    # convert the order of genes into the template format
    print("\nReordering methylation data to comply with template...")
    _prepare_data_file(
        METHY_FILE,
        METHY_OUT,
        gene_index=gene_index,
        sample_index=idx_samples_methy,
        sep=",",
    )

    # open gene expression data, remove all samples that do not have methylation data, and
    # reorder the genes to comply with the template
    print("\nReordering gene expression data to comply with template...")
    _prepare_data_file(
        EXP_FILE, EXP_OUT, gene_index=gene_index, sample_index=idx_samples_exp, sep=" "
    )


def create_dummy_sets(num_samples, dimension):
    """Script to create dummy datasets with custom dimensionality and sample size.
    """
    data_file = f"dummyset{num_samples}.csv"
    label_file = f"dummyset{num_samples}_labels.csv"

    # open data and label file for writing
    with open(DUMMY + data_file, "w") as f_data:
        with open(DUMMY + label_file, "w") as f_label:
            # create ids for each sample
            ids = [f"sample{i}" for i in range(num_samples)]

            # prepare label file
            for id in ids:
                if random.uniform(0, 1) > 0.5:
                    f_label.write(f"{id},True\n")
                else:
                    f_label.write(f"{id},False\n")

            # prepare data file
            # step 1: write id line
            f_data.write(",".join(ids) + "\n")

            # step 2: generate values for each feature
            for i in range(dimension):
                features = np.random.default_rng().uniform(5.0, 15.0, num_samples)
                f_data.write(
                    f"feature{i}," + ",".join(np.char.mod("%f", features)) + "\n"
                )


def analyze_runtime(runtime_path: str):
    """Script to analyze the runtime using dummysets of different size.
    """
    # define the sizes and filename template of dummysets
    set_sizes = [100, 1000, 10000, 100000]
    attention = ["none", "gated"]
    batch_size = ["fix", "adaptive"]
    temp_file = "dummyset{}_{}_{}_timings.pkl"

    time = {
        "none": {
            "fix": {"epoch": [], "time": []},
            "adaptive": {"epoch": [], "time": []},
        },
        "gated": {
            "fix": {"epoch": [], "time": []},
            "adaptive": {"epoch": [], "time": []},
        },
    }
    colors = {
        "none_fix": "b*-",
        "none_adaptive": "r*-",
        "gated_fix": "g*-",
        "gated_adaptive": "y*-",
    }

    # iterate over all conditions
    for s in set_sizes:
        for a in attention:
            for b in batch_size:
                if b == "fix":
                    filepath = runtime_path + temp_file.format(s, a, 32)
                else:
                    filepath = runtime_path + temp_file.format(s, a, int(s / 100))

                try:
                    time_dict = torch.load(filepath)

                    time[a][b]["epoch"].append(
                        [
                            time
                            for time_list in time_dict["times_epoch"]
                            for time in time_list
                        ]
                    )
                    time[a][b]["time"].append(
                        sum(time_dict["times"]) / len(time_dict["times"])
                    )

                except FileNotFoundError:
                    time[a][b]["time"].append(time[a][b]["time"][-1] + 5)
                    print(f"[Warning] No such file or directory: {filepath}")

    # visualize the times
    plt.figure()
    for a in attention:
        for b in batch_size:
            plt.plot(
                set_sizes,
                time[a][b]["time"],
                colors[f"{a}_{b}"],
                ms=10,
                label=f"{a}_{b}",
            )
    # plt.xticks(set_sizes, labels=[str(s) for s in set_sizes])
    plt.xlabel("Number of Samples")
    plt.ylabel("Mean Training Time (min)")
    # plt.xscale("log")
    # plt.yscale("log")
    plt.legend(loc="best")
    plt.show()


def analyze_runtime_multi(runtime_path: str):
    """Script to analyze the computational effiziency of COmic with increasing
       number of omics modalities.
    """
    # define needed variables
    nb_omics = [2, 3, 4]
    filename = "{}_gated_timings.pkl"
    time_train = []

    # iterate over all multi omics cases
    for nbo in nb_omics:
        filepath = runtime_path + filename.format(nbo)
        
        # load timing results
        try:
            time_dict = torch.load(filepath)
            time_train.append(
                sum(time_dict["times"]) / len(time_dict["times"])
            )

        except FileNotFoundError:
            time_train.append(-1)
            print(f"[Warning] No such file or directory: {filepath}")

    # visualize the times
    plt.figure()
    plt.plot(nb_omics, time_train, "g*-", ms=10)
    plt.xlabel("Number of omics Modalities")
    plt.ylabel("Mean Training Time (min)")
    plt.show()


def analyze_cunetal(network: Tuple, run: str):
    """Script to analyze and compare the performance of Comic Networks on Cunetal data.
    """

    def load_baseline():
        """Function to load baseline results
        """
        baseline = pd.read_csv(BASELINE)
        baselines = baseline.groupby(baseline.index)

        bl_dict = {}
        for name, values in baselines:
            aux_values = values.to_numpy()
            bl_dict[name] = aux_values.flatten()

        return bl_dict

    def load_pimkl():
        """Function to load the PIMKL results
        """
        sets = ["GSE11121", "GSE1456", "GSE2034", "GSE2990", "GSE4922", "GSE7390"]

        pimkl_results = np.zeros((len(sets), 100))
        pimkl_results_mean = np.zeros((len(sets), 10))
        for i, dataset in enumerate(sets):
            df_res = pd.read_csv(PIMKL_CUNETAL.format(dataset, dataset))
            pimkl_results[i, :] = df_res["Relapse<5Years"].to_numpy().flatten()

            for j in range(10):
                pimkl_results_mean[i, j] = np.mean(
                    pimkl_results[i, j * 10 : j * 10 + 9]
                )

        return pimkl_results.flatten(), pimkl_results_mean.flatten()

    # load the baseline and PIMKL results
    baselines = load_baseline()
    _, pimkl_res_mean = load_pimkl()

    # load the kernel network results
    res_path = RESULT.format(network[0], run)
    sets = ["GSE11121", "GSE1456", "GSE2034", "GSE2990", "GSE4922", "GSE7390"]

    more_than_one_model = any(
        [os.path.isdir(res_path + i) for i in os.listdir(res_path)]
    )
    if more_than_one_model:
        comik_results = []
        comik_results_mean = []
        comik_labels = []
        for mtype in os.listdir(res_path):
            if os.path.isdir(res_path + mtype):
                aux_res_path = res_path + f"{mtype}/"
                aux_comik_results = np.zeros((len(sets), 100))
                aux_comik_results_mean = np.zeros((len(sets), 10))
                for i, dataset in enumerate(sets):
                    df_res = pd.read_csv(
                        aux_res_path + f"{dataset}_{network[1]}_all.csv"
                    )
                    aux_comik_results[i, :] = df_res.loc[4, :].to_numpy()[1:]

                    for j in range(10):
                        aux_comik_results_mean[i, j] = np.mean(
                            aux_comik_results[i, j * 10 : j * 10 + 9]
                        )
                comik_results.append(aux_comik_results)
                comik_results_mean.append(aux_comik_results_mean)
                comik_labels.append(f"COmiK_{mtype}")

    else:
        comik_results = np.zeros((len(sets), 100))
        comik_results_mean = np.zeros((len(sets), 10))
        for i, dataset in enumerate(sets):
            df_res = pd.read_csv(res_path + f"{dataset}_{network[1]}_all.csv")
            comik_results[i, :] = df_res.loc[4, :].to_numpy()[1:]

            for j in range(10):
                comik_results_mean[i, j] = np.mean(
                    comik_results[i, j * 10 : j * 10 + 9]
                )

    # ensure that the ordering of the boxplots is identical to the origial PIMKL manuscript
    boxes = []
    labels = []
    for bl in [
        "PAM",
        "sigGenNB",
        "sigGenSVM",
        "SCAD",
        "HHSVM",
        "RFE",
        "RRFE",
        "graphK",
        "graphKp",
        "networkSVM",
        "PAC",
        "aveExpPath",
        "HubClassify",
        "pathBoost",
    ]:
        boxes.append(baselines[bl])
        labels.append(bl)
    boxes.append(pimkl_res_mean)
    labels.append("PIMKL")

    # add Comik results
    if more_than_one_model:
        for i, mtype in enumerate(comik_labels):
            boxes.append(comik_results_mean[i].flatten())
            labels.append(mtype)
    else:
        boxes.append(comik_results_mean.flatten())
        labels.append("COmiK")

    # calculate mean and std for each model
    mean_models = [np.mean(i) for i in boxes]
    std_models = [np.std(i) for i in boxes]
    print("AUC (mean +- std) for each model:")
    for i in range(len(boxes)):
        print(f"    {labels[i]}: {mean_models[i]} +- {std_models[i]}")

    # style configs
    black_diamond = dict(markerfacecolor="k", marker="D")
    box_style = dict(linewidth=2.5, color="darkblue")
    median_style = dict(linewidth=2.5, color="firebrick")
    whiskers_style = dict(linewidth=2.5, color="darkblue")

    plt.figure()
    bplot = plt.boxplot(
        boxes,
        labels=labels,
        notch=True,
        bootstrap=10000,
        patch_artist=True,
        flierprops=black_diamond,
        boxprops=box_style,
        medianprops=median_style,
        whiskerprops=whiskers_style,
        capprops=whiskers_style,
    )
    plt.xticks(rotation=45, ha="right")

    # highlight Comik results
    # bplot["boxes"][-1].set_linecolor("orange")
    bplot["boxes"][-1].set(facecolor="orange", edgecolor="darkorange")
    bplot["whiskers"][-1].set(color="darkorange")
    bplot["caps"][-1].set(color="darkorange")
    bplot["whiskers"][-2].set(color="darkorange")
    bplot["caps"][-2].set(color="darkorange")
    bplot["boxes"][-2].set(facecolor="orange", edgecolor="darkorange")
    bplot["whiskers"][-3].set(color="darkorange")
    bplot["caps"][-3].set(color="darkorange")
    bplot["whiskers"][-4].set(color="darkorange")
    bplot["caps"][-4].set(color="darkorange")

    plt.show()


def analyze_metabric(network: Tuple, run: str):
    """Script to compare omics network performance on METABRIC data
    to PIMKL.
    """

    def load_pimkl(baselines):
        """Function to load the PIMKL results
        """

        pimkl_results = []
        pimkl_results_mean = []
        for baseline in baselines:

            if baseline == "cna+mrna":
                filepath = PIMKL_METABRIC.format(baseline, "kegg-pc+kegg-pc")
            else:
                filepath = PIMKL_METABRIC.format(baseline, "kegg-pc")

            df_res = pd.read_csv(filepath)
            aux_res = df_res["class"].to_numpy().flatten()

            aux_res_mean = np.zeros(10)
            for j in range(10):
                aux_res_mean[j] = np.mean(aux_res[j * 10 : j * 10 + 9])
            pimkl_results.append(aux_res)
            pimkl_results_mean.append(aux_res_mean)

        return pimkl_results, pimkl_results_mean

    baselines = ["cna", "mrna", "cna+mrna"]

    # load PIMKL performance
    pimkl_result, _ = load_pimkl(baselines)

    # load Comik results
    comik_result = []
    for b in baselines:
        # specify the correct file that holds the results
        if b == "cna+mrna":
            res_file = (
                RESULT.format(network[0], run) + f"MultiOmics_multi{network[1]}_all.csv"
            )
        else:
            res_file = RESULT.format(network[0], run) + f"{b}_{network[1]}_all.csv"

        # load results
        df_res = pd.read_csv(res_file)
        comik_result.append(df_res.loc[4, :].to_numpy()[1:])

    # plot results
    def set_box_color(bp, colors):
        for i, c in enumerate(colors):
            if c == "darkorange":
                plt.setp(bp["boxes"][i], facecolor="orange", edgecolor=c, linewidth=2.5)
            else:
                plt.setp(
                    bp["boxes"][i], facecolor="steelblue", edgecolor=c, linewidth=2.5
                )
            plt.setp(bp["medians"][i], color="firebrick", linewidth=2.5)
            plt.setp(bp["whiskers"][i * 2], color=c, linewidth=2.5)
            plt.setp(bp["whiskers"][i * 2 + 1], color=c, linewidth=2.5)
            plt.setp(bp["caps"][i * 2], color=c, linewidth=2.5)
            plt.setp(bp["caps"][i * 2 + 1], color=c, linewidth=2.5)

    plt.figure()

    # draw boxes
    bp_pimkl = plt.boxplot(
        pimkl_result,
        positions=np.arange(len(pimkl_result)) * 2.0 - 0.4,
        sym="",
        widths=0.6,
        notch=True,
        bootstrap=10000,
        patch_artist=True,
    )
    bp_comik = plt.boxplot(
        comik_result,
        positions=np.arange(len(comik_result)) * 2.0 + 0.4,
        sym="",
        widths=0.6,
        notch=True,
        bootstrap=10000,
        patch_artist=True,
    )

    # color boxes
    set_box_color(bp_pimkl, ["darkblue", "darkblue", "darkblue"])
    set_box_color(bp_comik, ["darkorange", "darkorange", "darkorange"])

    # prepare axes
    plt.xticks(np.arange(0, len(baselines) * 2, 2), baselines)
    plt.xlim(-2, len(baselines) * 2)
    plt.tight_layout()
    plt.title("Metabric")
    plt.show()


def analyze_bic(network, run):
    """Script to vizualize the results on breast invasive carcinoma
    data.
    """
    baselines = ["exp", "methy", "MultiOmics"]

    # load Comik results
    comik_result = []
    for b in baselines:
        # specify the correct file that holds the results
        if b == "MultiOmics":
            res_file = RESULT.format(network[0], run) + f"{b}_multi{network[1]}_all.csv"
        else:
            res_file = RESULT.format(network[0], run) + f"{b}_{network[1]}_none_all.csv"

        # load results
        df_res = pd.read_csv(res_file)
        comik_result.append(df_res.loc[4, :].to_numpy()[1:])

    # style configs
    black_diamond = dict(markerfacecolor="k", marker="D")
    box_style = dict(linewidth=2.5, color="darkblue")
    median_style = dict(linewidth=2.5, color="firebrick")
    whiskers_style = dict(linewidth=2.5, color="darkblue")

    plt.figure()
    plt.boxplot(
        comik_result,
        labels=baselines,
        notch=True,
        bootstrap=10000,
        patch_artist=True,
        flierprops=black_diamond,
        boxprops=box_style,
        medianprops=median_style,
        whiskerprops=whiskers_style,
        capprops=whiskers_style,
    )
    plt.xticks(rotation=45, ha="right")
    plt.title("BIC")
    plt.show()


def load_pathways(dataset):
    """Script to load pathway names given a dataset
    """
    # load the pathway names
    pathways = []
    directory = os.fsencode(PATHWAYS.format(dataset))
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        pathways.append(" ".join(filename.split("_")[1:-4]).upper())

    return pathways


def load_pathway_weights_pool(network, run, dataset, model):
    """Load the weight of each pathway of a trained model
    """
    filepath = RESULT + "{}_{}_{}.pkl"

    # load results and retrieve the weigths of the fully connected layer
    res_dict = torch.load(filepath.format(network, run, dataset, model, "none"))
    weights = res_dict["state_dict"]["fc.weight"].cpu().numpy()

    return weights[0, :]


def analyze_pathway_weights_cunetal(network, run, model):
    """Script to analyze the influence of different pathways
    """
    # specify the datasets
    datasets = ["GSE11121", "GSE1456", "GSE2034", "GSE2990", "GSE4922", "GSE7390"]

    # load the names of the pathways
    pathways = load_pathways(datasets[0])

    # load the weights of the models
    weights = []
    for d in datasets:
        weights.append(load_pathway_weights_pool(network, run, d, model))

    # select the plot type
    plottype = "box"

    if plottype == "line":
        # define color for each line plot
        linecolors = [
            "gold",
            "steelblue",
            "darkorange",
            "darkmagenta",
            "firebrick",
            "forestgreen",
        ]

        # plot the weight distributions
        plt.figure()
        for i, d in enumerate(datasets):
            plt.plot(weights[i], label=d, color=linecolors[i], linewidth=2)
        plt.xticks(
            ticks=np.arange(len(pathways)), labels=pathways, rotation=45, ha="right"
        )
        plt.legend(loc="best")
        plt.show()

    elif plottype == "box":
        # convert weight lines into boxes
        boxes_weight = []
        for i in range(len(weights[0])):
            # get weight of each model for current pathway
            aux_box = []
            for j in range(len(weights)):
                aux_box.append(weights[j][i])
            boxes_weight.append(aux_box)

        # define the box style
        black_diamond = dict(markerfacecolor="k", marker="D")
        box_style = dict(linewidth=2.5, edgecolor="k", facecolor="grey")
        median_style = dict(linewidth=2.5, color="firebrick")
        whiskers_style = dict(linewidth=2.5, color="k")

        # prepare plot
        plt.figure()
        plt.boxplot(
            boxes_weight,
            patch_artist=True,
            showfliers=False,
            flierprops=black_diamond,
            boxprops=box_style,
            medianprops=median_style,
            whiskerprops=whiskers_style,
            capprops=whiskers_style,
        )
        # plt.axhline(y=0.4, color="k", linestyle="--")
        # plt.axhline(y=-0.4, color="k", linestyle="--")
        # plt.axhline(y=0.0, color="k", linestyle="--")
        plt.show()


def _viz_gridsearch_attention(datasets, network, run, anchors, attention, adim):
    """Visualize gridsearch results for attention networks"""

    # iterate over each dataset
    for d in datasets:
        # iterate over all grid search results and store them
        res = []
        labels = []
        for at in attention:
            for an in anchors:
                for ad in adim:
                    # generate the path to the result file
                    #   -> if you used different datasets, please
                    #      make the appropriate changes to the filename variable
                    filename = f"attention/adim_{ad}/{d}_{network[1]}_{at}_{an}_all.csv"
                    filepath = RESULT.format(network[0], run) + filename

                    # load the results
                    try:
                        df_res = pd.read_csv(filepath)
                        res.append(df_res.loc[4, :].to_numpy()[1:])
                    except FileNotFoundError:
                        df_res = []
                        res.append(np.array(df_res))
                    labels.append(f"{at}, {an} anchors, {ad} adim")

        # vizualize the results
        # style configs
        black_diamond = dict(markerfacecolor="k", marker="D")
        box_style = dict(linewidth=2.5, color="darkblue")
        median_style = dict(linewidth=2.5, color="firebrick")
        whiskers_style = dict(linewidth=2.5, color="darkblue")

        plt.figure()
        plt.boxplot(
            res,
            labels=labels,
            notch=True,
            patch_artist=True,
            flierprops=black_diamond,
            boxprops=box_style,
            medianprops=median_style,
            whiskerprops=whiskers_style,
            capprops=whiskers_style,
        )
        plt.xticks(rotation=45, ha="right", fontsize=5)
        plt.title(f"{d}")
        plt.show()


def _viz_gridsearch_pooling(datasets, network, run, anchors):
    """Visualize gridsearch results for pooling networks"""
    # iterate over each dataset
    for d in datasets:
        # iterate over all grid search results and store them
        res = []
        labels = []
        for an in anchors:
            # generate the path to the result file
            #   -> if you used different datasets, please
            #      make the appropriate changes to the filename variable
            filename = f"pool/{d}_{network[1]}_none_{an}_all.csv"
            filepath = RESULT.format(network[0], run) + filename

            # load the results
            try:
                df_res = pd.read_csv(filepath)
                res.append(df_res.loc[4, :].to_numpy()[1:])
            except FileNotFoundError:
                df_res = []
                res.append(np.array(df_res))
            labels.append(f"{an} anchors")

        # vizualize the results
        # style configs
        black_diamond = dict(markerfacecolor="k", marker="D")
        box_style = dict(linewidth=2.5, color="darkblue")
        median_style = dict(linewidth=2.5, color="firebrick")
        whiskers_style = dict(linewidth=2.5, color="darkblue")

        plt.figure()
        plt.boxplot(
            res,
            labels=labels,
            notch=True,
            patch_artist=True,
            flierprops=black_diamond,
            boxprops=box_style,
            medianprops=median_style,
            whiskerprops=whiskers_style,
            capprops=whiskers_style,
        )
        plt.xticks(rotation=45, ha="right", fontsize=5)
        plt.title(f"{d}")
        plt.show()


def viz_gridsearch(network, run):
    """This function visualizes the results of a gridsearch
    """
    # define the datasets for which the gridsearch results should be visualized
    #   -> change this if you want to display gridsearch results for different datasets
    DATASETS = [
        "exp",
        "methy",
    ]  # ["GSE11121", "GSE1456", "GSE2034", "GSE2990", "GSE4922", "GSE7390"]

    # define gridsearch parameters
    #   -> change this if you selected different parameters during the gridsearch
    model_type = ["pooling"]  # ["pooling", "attention"]
    anchors = [10, 20, 30, 40]
    attention = ["normal", "gated"]
    adim = [32, 64, 128]

    for mtype in model_type:
        if mtype == "pooling":
            _viz_gridsearch_pooling(DATASETS, network, run, anchors)

        elif mtype == "attention":
            _viz_gridsearch_attention(DATASETS, network, run, anchors, attention, adim)


def _read_in_laplacians(dirpath: str, header=None, sep: str = ","):
    """Function to read in predefined Laplacians."""
    # encode directory path in byte
    directory = os.fsencode(dirpath)

    # iterate through all files in the specified folder
    pi_laplacians = []
    for file in os.listdir(directory):

        # decode the filepath from bytes to characters
        filename = os.fsdecode(os.path.join(directory, file))

        # read in the laplacian with pandas
        laplacian = pd.read_csv(filename, header=header, sep=sep)
        laplacian.columns = ["row", "column", "data"]

        # remove all the zero rows
        laplacian = laplacian.loc[laplacian["data"] != 0]

        # get the nodes that are included in the current laplacian
        node_ids = np.array(laplacian["row"].unique())

        # create the laplacian that contains only the relevant nodes
        l_matrix = np.zeros((len(node_ids), len(node_ids)))
        for _, row in laplacian.iterrows():
            l_matrix[
                np.where(node_ids == row["row"]), np.where(node_ids == row["column"])
            ] = row["data"]

        pi_laplacians.append((node_ids, l_matrix))

    return pi_laplacians


def get_local_interpretation(modeltype, run, datapath):
    """This function loads a model and calculates the local interpretation
    for the samples provided in the datafile.
    CAUTION: Only works for attention-based COmic models"""
    # load the model parameters
    dataset = "GSE11121"
    aux_path = RESULT + "{}_{}_gated.pkl"
    res_dict = torch.load(aux_path.format(modeltype[0], run, dataset, modeltype[1]))

    # check which model type is used
    if modeltype[1] == "pimkl":
        # load omics data
        data_all = OmicsDataset(
            data_file=datapath[0],
            label_file=datapath[1],
            omics_format=res_dict["args"].omics_format,
            class_labels=res_dict["args"].classes,
            raise_unequal_sample_count=False,
        )

        # load the used laplacians
        pi_laplacians = _read_in_laplacians(res_dict["args"].laplacians)

        # initialize a model
        model = PIMKLNet(
            num_features=data_all.num_features,
            num_classes=res_dict["args"].num_classes,
            num_anchors=res_dict["args"].num_anchors,
            pi_laplacians=pi_laplacians,
            pooling=res_dict["args"].pooling,
            attention=res_dict["args"].attention,
            attention_params=res_dict["args"].attention_params,
        )

        # load trained model parameters
        model.load_state_dict(res_dict["state_dict"])
    elif modeltype[1] == "multipimkl":
        # load multi-omics data
        data_all = MultiOmicsDataset(
            data_files=datapath[:-1],
            label_file=datapath[-1],
            omics_format=res_dict["args"].omics_format,
            class_labels=res_dict["args"].classes,
            raise_unequal_sample_count=False,
        )

        # load the used laplacians
        pi_laplacians = []
        for l_path in res_dict["args"].laplacians:
            pi_laplacians.append(_read_in_laplacians(l_path))

        # initialize a model
        model = MultiPIMKLNet(
            num_features=data_all.num_features,
            num_classes=res_dict["args"].num_classes,
            num_anchors=res_dict["args"].num_anchors,
            pi_laplacians=pi_laplacians,
            pooling=res_dict["args"].pooling,
            attention=res_dict["args"].attention,
            attention_params=res_dict["args"].attention_params,
        )

        # load trained model parameters
        model.load_state_dict(res_dict["state_dict"])
    else:
        raise ValueError(f"Selected modeltype {modeltype} is not supported")

    # create a DataLoader
    data_loader = DataLoader(data_all, batch_size=1, shuffle=False)

    # make prediction on the dataset
    y_pred, y_true = model.predict(data_loader, proba=True, show_attention_weight=True)

    # concatenate predicted and true labels
    labels = torch.zeros((len(y_pred), 2))
    labels[:, 0] = y_true
    labels[:, 1] = y_pred

    # get samples that were correctly classified as true, correctly classified as false,
    # and wrongly classified as true
    correct_zero = torch.mul(labels[:, 0] < 0.5, labels[:, 1] < 0.5).nonzero()
    correct_one = torch.mul(labels[:, 0] > 0.5, labels[:, 1] > 0.5).nonzero()
    wrong_zero = torch.mul(labels[:, 0] < 0.5, labels[:, 1] > 0.5).nonzero()
    wrong_one = torch.mul(labels[:, 0] > 0.5, labels[:, 1] < 0.5).nonzero()

    # make sure that the whole tensor is printed out
    print("\n\n=======\nLabels\n=======\n")
    torch.set_printoptions(edgeitems=1000)
    print(labels)
    print("Correct Zeros: ", correct_zero.flatten())
    print("Correct Ones: ", correct_one.flatten())
    print("Wrong Zeros: ", wrong_zero.flatten())
    print("Wrong Ones: ", wrong_one.flatten())

    # randomly select one of the samples specified above
    torch.manual_seed(42)
    print("\n\n==============\nRandom Indices\n==============\n")
    if len(correct_zero) > 0:
        print(
            f"Correct Zero: {correct_zero[torch.randint(len(correct_zero), (1,))].item()}"
        )
    if len(correct_one) > 0:
        print(
            f"Correct One: {correct_one[torch.randint(len(correct_one), (1,))].item()}"
        )
    if len(wrong_zero) > 0:
        print(f"Wrong Zero: {wrong_zero[torch.randint(len(wrong_zero), (1,))].item()}")
    if len(wrong_one) > 0:
        print(f"Wrong One: {wrong_one[torch.randint(len(wrong_one), (1,))].item()}")


def _viz_local():
    """Vizualization routine for local interpretations. Please get the attention weights using the
    terminal functionality of this script (i.e. --type local). Afterwards the weights should be
    copied into this function to generate the plot.
    """
    RIGHT_ZERO = [
        4.5320e-04,
        3.9561e-04,
        6.3124e-02,
        6.3124e-02,
        8.8335e-04,
        8.8365e-04,
        3.0209e-06,
        1.8708e-07,
        1.0418e-03,
        6.3124e-02,
        8.8342e-04,
        6.5011e-04,
        3.4415e-04,
        3.0044e-04,
        6.3124e-02,
        8.3743e-02,
        8.8341e-04,
        6.3124e-02,
        2.0106e-03,
        8.8342e-04,
        8.2901e-04,
        3.9557e-04,
        8.8342e-04,
        8.5829e-05,
        5.3552e-04,
        8.8343e-04,
        9.6273e-08,
        6.3124e-02,
        3.2020e-04,
        7.7937e-02,
        6.3124e-02,
        8.7115e-04,
        2.1642e-04,
        8.8333e-04,
        6.3112e-02,
        8.8342e-04,
        8.0662e-05,
        6.3124e-02,
        3.9557e-04,
        3.9551e-04,
        7.3463e-02,
        6.3123e-02,
        8.8342e-04,
        2.8926e-04,
        8.7519e-04,
        2.9084e-04,
        6.3111e-02,
        8.8342e-04,
        2.9084e-04,
        4.9730e-02,
    ]
    RIGHT_ONE = [
        2.9830e-05,
        2.6040e-05,
        4.1548e-03,
        4.1548e-03,
        5.8144e-05,
        5.8157e-05,
        4.2965e-08,
        1.3202e-08,
        5.9159e-05,
        4.1548e-03,
        5.8147e-05,
        4.2970e-05,
        2.4470e-05,
        2.3108e-05,
        4.1548e-03,
        8.6084e-03,
        5.8147e-05,
        4.1548e-03,
        9.4382e-04,
        5.8146e-05,
        4.4031e-05,
        2.6037e-05,
        5.8147e-05,
        4.7630e-06,
        4.9256e-05,
        5.8147e-05,
        4.6574e-09,
        4.1548e-03,
        2.5959e-05,
        9.3519e-01,
        4.1548e-03,
        5.5275e-05,
        1.5366e-05,
        5.8140e-05,
        4.1540e-03,
        5.8147e-05,
        5.8452e-06,
        4.1548e-03,
        2.6037e-05,
        2.6037e-05,
        4.1986e-03,
        4.1548e-03,
        5.8147e-05,
        1.9131e-05,
        5.5127e-05,
        1.9154e-05,
        4.1512e-03,
        5.8147e-05,
        1.9148e-05,
        4.1226e-03,
    ]
    WRONG_ONE = [
        4.2619e-04,
        3.7155e-04,
        5.9363e-02,
        5.9363e-02,
        8.3022e-04,
        8.3105e-04,
        8.2266e-08,
        4.2967e-09,
        8.8274e-04,
        5.9362e-02,
        8.3078e-04,
        6.3052e-04,
        2.9787e-04,
        3.1122e-04,
        5.9363e-02,
        1.5463e-01,
        8.3075e-04,
        5.9363e-02,
        1.5275e-02,
        8.3077e-04,
        9.7789e-04,
        3.7200e-04,
        8.3066e-04,
        8.4125e-05,
        3.1575e-04,
        8.3082e-04,
        4.5336e-08,
        5.9363e-02,
        3.6057e-04,
        8.8934e-02,
        5.9363e-02,
        2.9446e-04,
        1.4071e-04,
        8.3079e-04,
        5.9355e-02,
        8.3078e-04,
        1.0080e-04,
        5.9363e-02,
        3.7200e-04,
        3.7198e-04,
        6.1886e-02,
        5.9362e-02,
        8.3078e-04,
        8.2183e-05,
        6.4827e-04,
        2.7347e-04,
        5.9329e-02,
        8.3078e-04,
        2.7565e-04,
        9.5998e-03,
    ]

    # create the heat map matrix and fill with numbers
    heatmap_matrix = np.zeros((3, len(RIGHT_ONE)))
    heatmap_matrix[0, :] = RIGHT_ZERO
    heatmap_matrix[1, :] = RIGHT_ONE
    heatmap_matrix[2, :] = WRONG_ONE

    # create plot
    fig, axs = plt.subplots(3)
    for i in range(3):
        im = axs[i].imshow(
            heatmap_matrix[i, :].reshape((1, 50)),
            cmap="hot",
            interpolation=None,
            aspect="auto",
        )
        fig.colorbar(im, ax=axs[i])
    plt.xticks(np.arange(50))
    plt.show()


def main():
    args = load_args()

    if args.type == "cunetal":
        analyze_cunetal(args.network, args.run)

    elif args.type == "metabric":
        analyze_metabric(args.network, args.run)

    elif args.type == "bic":
        analyze_bic(args.network, args.run)

    elif args.type == "time":
        runtime_path = RESULT.format(args.network[0], args.run)
        analyze_runtime(runtime_path)

    elif args.type == "multitime":
        runtime_path = RESULT.format(args.network[0], args.run)
        analyze_runtime_multi(runtime_path)

    elif args.type == "gridsearch":
        viz_gridsearch(args.network, args.run)

    elif args.type == "preprocessing":
        # prepare_multiomics()

        # if you have already preprocessed the multi-omics dataset and want to
        # count the number of samples in each class, comment out the call to
        # prepare_multiomics() and uncomment the following line
        # _count_bic_labels()
        _count_metabric_labels()

    elif args.type == "global":
        analyze_pathway_weights_cunetal(args.network[0], args.run, args.network[1])

    elif args.type == "local":
        get_local_interpretation(args.network, args.run, args.data)

        # If attantion weights are already obtained, uncommend the following line
        # and run the script with '--type local'. You have to comment out the
        # call to get_local_interpretation().
        # ATTENTION: Will only work if you have run 'python analysis.py --type local'
        #            once with the call to get_local_interpretation.
        # _viz_local()


if __name__ == "__main__":
    main()
