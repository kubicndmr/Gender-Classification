import os
import csv

def generate_labels(path):
    wavs = os.listdir(path)
    labels = list()
    
    for item in wavs:
        if item.startswith('f'):
            labels.append((item,'f'))
        elif item.startswith('m'):
            labels.append((item,'m'))
        
    with open(path+"labels.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(labels)
        
def load_labels(path):
    List,Label = list(), list()
    with open(path+'labels.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            List.append(row[0])
            Label.append(row[1])
    return List,Label
