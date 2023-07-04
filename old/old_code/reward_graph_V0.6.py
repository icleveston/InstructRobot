import matplotlib, csv
matplotlib.rcParams["backend"] = "TkAgg"
from matplotlib import pyplot as plt
from os.path import join
import numpy as np

filexp = []

# filexp.append("name-of-experiment_2020_08_19_15_43_07_0000--s-0")
# filexp.append("name-of-experiment_2020_08_20_01_49_20_0000--s-0")
# filexp.append("name-of-experiment_2020_08_16_03_04_27_0000--s-0")
# filexp.append("")
filexp.append("ExpFreeRise_2020_08_20_19_18_27_0000--s-0")

# fileroot = "/home/nanbaima/Documents/naovrepenv/rlkit/data/name-of-experiment/"
fileroot = "/home/nanbaima/Documents/naovrepenv/rlkit/data/ExpFreeRise/"

def plot_multivalue(values, columns, labels, title, mean, path):
    handles = []

    plt.clf()
#    plt.style.use('seaborn-white')
    for c, l in zip(columns, labels):
        v = [float(v[c]) for v in values]
        v_mean=np.mean(np.array(v[:len(v)-len(v)%mean]).reshape(-1, mean), axis=1)
        handles.append(plt.plot(range(len(v_mean)), v_mean, label=l)[0])

    plt.legend(handles=handles)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path + "/" + title + ".png", dpi=800)
    # plt.show()
    # plt.close()

if __name__ == "__main__":
	for i in range(len(filexp)):
		filedic = join(fileroot, filexp[i])
		f = open(join(filedic,"reward.csv"))
		reader = csv.reader(f, delimiter=",")
		values = [v for v in reader]
		#values = [line[:1] + [10 if float(line[1]) < 0.1 else 0] + line[2:] for 
		#           line in values]
		# for v in values:
		#     v.append(10)

		plot_multivalue(
		    values,
		    [0, 1, 2,
		    3,
		    4,
		    5, 6,
		    7,
		    # 8, 9],
		    # 6, 7, 8],
		    # 7, 8, 9],
		    # 3, 4, 5],
		    8, 9, 10],
		    ["3 collisions", "Touch", "No Touch", 
		      "Ball Caught",
		    "Open Hand",
		      "Ball Grabbed", "Ball Let Go",
		    "Ball Raised",
		      "Lef Hand Distance", "Right Hand Distance", 
		      "Reward"],
		    "Described Reward",
		    5000,
		    filedic)
