import os

file1 = open("nus_sms-data.txt", "r")
file2 = open("ovv.txt", "w")

i = 1

#short_line = open("nus_sms-data.txt","r").read()

for line in file1:
    if i%2==1:
        file2.write(line)

    i+=1

file1.close()
file2.close()
