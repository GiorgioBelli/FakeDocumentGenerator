#! /usr/bin/python
out_file = open("../../../FDG_Data/Results/generation/csvs/corpus_new_ML_MALWARE.csv", mode="w")

with open("../../../FDG_Data/Results/generation/csvs/corpus_ML_MALWARE.csv", mode="r",encoding="utf-8",errors="ignore") as csv_fd:

    lines = csv_fd.readlines()
    new_str = ""
    for line in lines:
        new_str += line[:-1] + "\f\n"

out_file.write(new_str)
out_file.close()
