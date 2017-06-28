import csv
from os import listdir
#change location to csv folder
with open('./HumanConcensus/Packet2Concensus.csv', 'w') as csvfile:
    fieldnames = ['file_name', 'Is_Cyberbullying_present']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    #change location to text folder location
    text_files = [f for f in listdir('./textpacket2')]
    for i in text_files:
    	writer.writerow({'file_name': i, 'Is_Cyberbullying_present': 'N'})
    