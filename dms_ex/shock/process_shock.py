from matplotlib import pyplot as plt
import numpy as np

import argparse
parser = argparse.ArgumentParser(
                    prog='process_shock',
                    description='Process SPARTA dump file',
                    epilog='Text at the bottom of help')

parser.add_argument("-f", "--file" )

args = parser.parse_args()

data = np.loadtxt(args.file, skiprows=118)

massrho= data[:,1]
nrho=data[:,2]
u=data[:,3]
v=data[:,4]
w=data[:,5]
temp= data[:,6]

fig, ax = plt.subplots()
ax.plot(temp)
fig.savefig("T.png")

fig, ax = plt.subplots()
ax.plot(u)
fig.savefig("U.png")

fig, ax = plt.subplots()
ax.plot(massrho)
fig.savefig("rho.png")
