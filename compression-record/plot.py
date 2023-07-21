#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input file')
    parser.add_argument('-o', '--output', help='output file')

    args = parser.parse_args()
    input_file = args.input
    output_file = args.output


    with open(input_file, 'r') as f:
        plots = csv.reader(f, delimiter=' ')
        x = []
        y = []
        for row in plots:
            x.append(float(row[0]))
            y.append(float(row[1]))



    plt.plot(x, y, label=input_file)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.yscale("linear")
    #plt.tick_params(bottom = False)
    #plt.tick_params(left= False)
    #plt.tick_params(labelbottom=False)
    #plt.tick_params(labelleft=False)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    print('Saving plot to {}'.format(output_file))
    plt.savefig(output_file)

