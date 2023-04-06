#%%
import glob

files = glob.glob("./bilateral-connectome/results/figs/*/*.svg")
start = "https://raw.githubusercontent.com/neurodata/bilateral-connectome/main/results/figs"

output = ""
for file in files:
    splits = file.split("/")[-2:]
    new_file = start + "/" + splits[0] + "/" + splits[1]
    new_output = "---\n\n![center](" + new_file + ")\n\n"
    output += new_output

print(output)