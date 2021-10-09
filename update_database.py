import os
arr = os.listdir("./static/Images")
print(arr)

os.system("rm podcast.csv")

f_podcast = open('podcast.csv','w')

for i in arr :

    if(i[-4:] == ".txt"):

        print(i)
        f = open("./static/Images/"+i,"r")

        data = f.read()
        
        data_replaced = data.replace("\n","")
        #data_replaced = data_replaced.replace("(","")
        #data_replaced = data_replaced.replace(")","")
        #data_replaced = data_replaced.replace("[","")
        #data_replaced = data_replaced.replace("]","")

        data_replaced_2 = data_replaced.replace(",","")

        if (int( len(data_replaced_2) / 1500 ) == 0):

            f_podcast.write("/" + f"A" + "/" + "," + f"{i}" + "," + "ABCD" + "," + data_replaced_2 + "," + "ENGLISH" + "\n")

        for j in range(0,int( len(data_replaced_2) / 1500 )):


            if( j >= int( len(data_replaced_2) / 1500 ) - 1 ) :

                data_replaced_truncated = data_replaced_2[ (j*1500) : ]

            else :

                data_replaced_truncated = data_replaced_2[ (j*1500) : 1500 + (j*1500) ]

            f_podcast.write("/" + f"A" + "/" + "," + f"{i}" + "," + "ABCD" + "," + data_replaced_truncated + "," + "ENGLISH" + "\n")
            j= j + 1

f_podcast.close()
f.close()

os.system("rm ./lyrics-data/temp.csv")
os.system("cp ./lyrics-data/podcast.csv ./lyrics-data/temp.csv")
os.system("cat podcast.csv >> ./lyrics-data/temp.csv")
os.system("cp ./lyrics-data/temp.csv ./lyrics-data/lyrics-toy-data1000.csv")
