import fileinput
import sys

'''
We use this file to replace the allocated CPU in the yaml file that defines the computeinfo pods.
'''

path = "/home/shahab/compute_info/scling_rules/"

def replace(a1, a2, a3, name):
    counter = 0
    for line in fileinput.input(name, inplace=True):
        if ("replicas:" in line):
            line = line.replace(line, '  replicas: ' + str(a1)) + "\n"
        if ("cpu:" in line):
            if (counter == 0):
                line = line.replace(line, '            cpu: "' + str(a2)) + 'm"' + "\n"
                counter += 1
            elif (counter == 1):
                line = line.replace(line, '            cpu: "' + str(a3)) + 'm"' + "\n"
                counter += 1
        sys.stdout.write(line)

print(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
replace(int(float(sys.argv[1])), int(float(sys.argv[2])),int(float(sys.argv[3])), path + sys.argv[4])