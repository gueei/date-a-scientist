import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

all_data = pd.read_csv("profiles.csv")

rate_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
all_data["drinks_code"] = all_data.drinks.map(rate_mapping)
smoke_mapping = {'sometimes': 1, 'no': 0, 'NaN': 0, 'when drinking': 1, 'yes': 2, 'trying to quit': 1}
all_data["smokes_code"] = all_data.smokes.fillna('NaN').map(smoke_mapping)
drug_mapping = {'never': 0, 'sometimes': 1, 'NaN': 0, 'often': 2}
all_data["drugs_code"] = all_data.drugs.fillna('NaN').map(drug_mapping)

essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
# Removing the NaNs
all_essays = all_data[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
all_data["essay_len"] = all_essays.apply(lambda x: len(x))


# like cats
all_data["pets"] = all_data["pets"].fillna("NA")
all_data["offspring"] = all_data["offspring"].fillna("NA")
all_data["education"] = all_data["education"].fillna("NA")

cat_mapping = {'likes dogs and likes cats': 1, 'has cats': 2, 'likes cats': 1, 'NA': 0,
       'has dogs and likes cats': 1, 'likes dogs and has cats': 2,
       'likes dogs and dislikes cats': -1, 'has dogs': 0,
       'has dogs and dislikes cats': -1, 'likes dogs': 0,
       'has dogs and has cats': 2, 'dislikes dogs and has cats': 2,
       'dislikes dogs and dislikes cats': -1, 'dislikes cats': -1,
       'dislikes dogs and likes cats': 1, 'dislikes dogs': 0}
pet_mapping = {'likes dogs and likes cats': 2, 'has cats': 2, 'likes cats': 1, 'NA': 0,
       'has dogs and likes cats': 3, 'likes dogs and has cats': 3,
       'likes dogs and dislikes cats': 0, 'has dogs': 2,
       'has dogs and dislikes cats': 1, 'likes dogs': 1,
       'has dogs and has cats': 4, 'dislikes dogs and has cats': 1,
       'dislikes dogs and dislikes cats': -2, 'dislikes cats': -1,
       'dislikes dogs and likes cats': 0, 'dislikes dogs': -1}
off_spring_mapping = {'doesn&rsquo;t have kids, but might want them': 1, 'NA': 0,
       'doesn&rsquo;t want kids': -1,
       'doesn&rsquo;t have kids, but wants them': 2,
       'doesn&rsquo;t have kids': 0, 'wants kids': 2, 'has a kid': 3, 'has kids': 4,
       'doesn&rsquo;t have kids, and doesn&rsquo;t want any': -1,
       'has kids, but doesn&rsquo;t want more': 3,
       'has a kid, but doesn&rsquo;t want more': 3,
       'has a kid, and wants more': 4, 'has kids, and might want more': 5,
       'might want kids': 1, 'has a kid, and might want more': 3,
       'has kids, and wants more': 5 }
education_mapping = {'working on college/university': 2.5, 'working on space camp': 2.5,
       'graduated from masters program': 3,
       'graduated from college/university':2, 'working on two-year college': 2,
       'NAN': 0, 'graduated from high school': 1, 'working on masters program': 2,
       'graduated from space camp': 2, 'college/university': 2,
       'dropped out of space camp': 1, 'graduated from ph.d program': 4,
       'graduated from law school': 2, 'working on ph.d program': 3.5,
       'two-year college': 1.5, 'graduated from two-year college': 1.5,
       'working on med school': 2, 'dropped out of college/university': 1,
       'space camp': 2, 'graduated from med school': 2,
       'dropped out of high school': 0, 'working on high school': 1.5,
       'masters program': 3, 'dropped out of ph.d program': 3,
       'dropped out of two-year college': 1, 'dropped out of med school': 1,
       'high school': 1, 'working on law school': 1.5, 'law school': 2,
       'dropped out of masters program': 2, 'ph.d program': 3.5,
       'dropped out of law school': 1.5, 'med school': 2}
sex_mapping = { 'm': 0, 'f': 1}
all_data["likes_cats"] = all_data.pets.map(cat_mapping)
all_data["likes_pets"] = all_data.pets.map(pet_mapping)
all_data["likes_children"] = all_data.offspring.map(off_spring_mapping)
all_data["education_level"] = all_data.education.map(education_mapping)
all_data["sex_code"] = all_data.sex.map(sex_mapping)




all_data.to_csv("profiles_processed.csv")