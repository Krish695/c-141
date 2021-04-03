from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as  np
df =pd.read_csv("final.csv")
df=df[df["soup"].notna()]
count=CountVectorizer(stop_words="english")
count_metrics=count.fit_transform(df["soup"])

df=df.reset_index()
indices=pd.Series(df.index,index=df["title"])
def get_recomendations(title):
    idx=indices[ title]
    sim_scores=list(enumerate(cosine_sim[idx]))
    sim_scores=sorted(sim_score,key=lambda x:x[1],reverse=True)
    sim_score=sim_scores[1:11]
    movie_indices=[i[0]for i in sim_score]
    return df[["title","poter_link","release_date","runtime","vote_average","overview"]].iloc [movie_indices].values.tolist()
    