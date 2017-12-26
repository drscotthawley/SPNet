#! /usr/bin/env python3

# Designed to parse CSV files from @achmorrison, containing JSON info of annotations of steelpan images

import pandas as pd
import json
import os
from shutil import copy2
import errno

# found it helpful to color the output since the strings were so long
from colorama import init
from colorama import Fore, Back, Style
init()

debug = 0

def make_sure_path_exists(path):
    try:                # go ahead and try to make the the directory
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:  # ignore error if dir already exists
            raise


def create_text_file(filename, datalist):
    if (debug > 2):
        print("   Creating text file ",filename,":")
    datastring = ""
    num_an = len(datalist)           # number of antinodes
    for i in range(num_an):
        if (debug > 3):
            print(" i = ",i,", an = ",an," datalist[i] = ",datalist[i])
        datastring += "[{0}, {1}, {2}, {3}, {4}, {5}]".format(datalist[i][0], datalist[i][1], datalist[i][2], datalist[i][3], datalist[i][4], datalist[i][5])
        if (i < num_an-1):
            datastring += "\n"

    with open(filename, "w") as text_file:
        text_file.write(datastring)
    return


outpath = 'parsed_zooniverze_steelpan'
inpath = 'zooniverse_steelpan'
make_sure_path_exists(outpath)

filename = 'zooniverse_labeled_dataset.csv'
df = pd.read_csv(filename)

for index, row in df.iterrows():
    line = row["annotations"]
    subject_id = row["subject_ids"]
    subject_data = row["subject_data"]
    user_name = row["user_name"]
    datalist = []
    problem = False
    sd_dict = json.loads(subject_data)
    sd_dict2 = sd_dict[str(subject_id)]
    ref_filename = sd_dict2["Filename"]
    ref_filename = ref_filename.replace('bmp.png','png')   # lets just use all .PNGs, k? thx
    #print(" user_name =[",user_name,"]",sep="")
    # TODO: for now we append the username to files we create; in the future it would be nice to "merge" or "combine" multiple users' work on the same image
    image_filename = os.path.splitext(ref_filename)[0]+"_"+user_name+'.png'
    txt_filename = os.path.splitext(ref_filename)[0]+"_"+user_name+'.txt'
    jlist = json.loads(line)
    if (debug>0):
        print("sd_dict = ",sd_dict)
        print(Fore.MAGENTA+"image_filename =",image_filename,", txt_filename = ",txt_filename)
        print(Fore.GREEN+"jlist=",jlist)
    if (len(jlist) < 2):
        print(Fore.RED+"   No antinodes in this image")
    else:
        i = 1   # skip i=0, which is header info
        dict_i = jlist[i]
        antinode_list = dict_i['value']
        num_an = len(antinode_list)
        if (debug > 2):
            print(Fore.YELLOW+"        i = ",i,", dict_i = jlist[i] = ",dict_i)
            print(Fore.WHITE+"                antinode_list = dict_i['value'] = ",antinode_list)
            print(           "                num_an = len(antinode_list) = ",num_an)  # number of antinodes
        for an in range(num_an):
            info = antinode_list[an]
            if (debug > 3):
                print(Fore.CYAN+"                   info = antinode_list[",an,"] = ",info)
            x = info['x']
            y = info['y']
            rx = info['rx']
            ry = info['ry']
            angle = info['angle']
            rings = info['details'][0]['value']
            # There are some problems in the data, e.g. value='' for 68115040 or value=None for 71179922. Ignore these images.
            if (rings != '') and (rings is not None):
                rings = int(rings)
                if (debug > 3):
                    print(Fore.CYAN+"                 Data:  x, y, rx, ry, angle, rings = ",x, y, rx, ry, angle, rings)
                datalist.append([x, y, rx, ry, angle, rings])
            else:
                print(Fore.RED+"   Problem.  Ring count unusable.  Ignoring whole image")
                problem = True
    if (False == problem):
        txt_filename = outpath+'/'+txt_filename
        if (debug>0):
            print(Fore.MAGENTA+"Output: Text file",txt_filename,": datalist = ",datalist)
        create_text_file(txt_filename,datalist)
        copy2(inpath+'/'+ref_filename, outpath+'/'+image_filename)
    if (debug > 0):
        print("")
        print("")

print(Fore.WHITE)
