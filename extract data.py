import csv
import math

def extract_data():
    file_open = open('India modified1.csv', 'r')
    f1 = csv.reader(file_open, delimiter=',')
    data1 = []
    for line in f1:
        data1.append(line)
    
    col_data = []
    for i in range(12): # 12 -> total number of years
        col_data.append([])
    col_k = 8 # column number
    for i in range(1,len(data1)):
        if data1[i][4] == 'MSSA': # Family / Phenotype
            if data1[i][1] == '2004': # Year
                col_data[0].append(float(data1[i][col_k]))
            elif data1[i][1] == '2005':
                col_data[1].append(float(data1[i][col_k]))
            elif data1[i][1] == '2006':
                col_data[2].append(float(data1[i][col_k]))
            elif data1[i][1] == '2007':
                col_data[3].append(float(data1[i][col_k]))
            elif data1[i][1] == '2008':
                col_data[4].append(float(data1[i][col_k]))
            elif data1[i][1] == '2009':
                col_data[5].append(float(data1[i][col_k]))
            elif data1[i][1] == '2015':
                col_data[6].append(float(data1[i][col_k]))
            elif data1[i][1] == '2016':
                col_data[7].append(float(data1[i][col_k]))
            elif data1[i][1] == '2018':
                col_data[8].append(float(data1[i][col_k]))
            elif data1[i][1] == '2019':
                col_data[9].append(float(data1[i][col_k]))
            elif data1[i][1] == '2020':
                col_data[10].append(float(data1[i][col_k]))
            elif data1[i][1] == '2021':
                col_data[11].append(float(data1[i][col_k]))
    
    for j in range(12): # Display count of records under each range of medicine quantity
        col_groups = [[], [], []]
        for i in col_data[j]:
            if i <= 2:
                col_groups[0].append(i)
            elif i == 4:# and i <= 8:
                col_groups[1].append(i)
            elif i >= 8:
                col_groups[2].append(i)
        print(len(col_groups[0]), len(col_groups[1]), len(col_groups[2]))

    return

extract_data()
