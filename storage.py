import csv 
all_movies=[]
with open ("final.csv") as f:

    csvreader=csv.reader(f)
    data=list(csvreader)
    all_movies=data[1:]
liked_movies=[]
not_liked_movies=[]
did_not_watch=[]