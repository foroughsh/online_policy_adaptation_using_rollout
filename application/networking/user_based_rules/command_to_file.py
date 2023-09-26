import fileinput
import sys

'''
:param: the input paramters to this function is the weights of the services and the address to the yaml file.

This function replace the values of the weights in configuration file with the given values. 

'''

path = "/home/shahab/compute_info/networking/user_based_rules/"

def replace_two_weight(a1,a2,name):
    if (a1>=0) and (a1<=100) and (a2>=0) and (a2<=100):
        counter = 0
        for line in fileinput.input(name, inplace=True):
            if ("weight:" in line):
                if (counter == 0):
                    line = line.replace(line, "      weight: " + str(a1)) + "\n"
                    counter += 1
                elif (counter == 1):
                    line = line.replace(line, "      weight: " + str(100 - a1)) + "\n"
                    counter += 1
                elif (counter == 2):
                    line = line.replace(line, "      weight: " + str(a2)) + "\n"
                    counter += 1
                elif (counter == 3):
                    line = line.replace(line, "      weight: " + str(100 - a2)) + "\n"
                    counter += 1
            sys.stdout.write(line)

def replace_one_weight(a1,name):
    if (a1>=0) and (a1<=100):
        counter = 0
        for line in fileinput.input(name, inplace=True):
            if ("weight:" in line):
                if (counter == 0):
                    line = line.replace(line, "      weight: " + str(a1)) + "\n"
                    counter += 1
                elif (counter == 1):
                    line = line.replace(line, "      weight: " + str(100 - a1)) + "\n"
                    counter += 1
            sys.stdout.write(line)

if (len(sys.argv) > 2):
    replace_two_weight(int(sys.argv[1]),int(sys.argv[2]),path + "user_based_weight.yaml")
else:
    replace_two_weight(int(sys.argv[1]), path + "user_based_weight.yaml")