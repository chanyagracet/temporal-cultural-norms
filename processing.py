# %% [markdown]
# ### Agglomerative clustering using SBERT

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import transformers
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

import pickle

# %%
# !pip install sentence_transformers

# %%
from sentence_transformers import SentenceTransformer

# %% [markdown]
# #### Loading the gpt generated files

# %%
# TODO: (if necessary) change the path to where GPT results are 
folder_path_name = "gpt_data/"

all_bolly = folder_path_name + "entire_bollywood_gpt-4o.csv"
all_holly = folder_path_name + "entire_hollywood_gpt-4o.csv"

all_bolly_df = pd.read_csv(all_bolly)
all_holly_df = pd.read_csv(all_holly)

# %% [markdown]
# ### Process the data to be in desired format

# %%
all_holly_df.columns

# %%
all_bolly_df.columns

# %%
def clean_decade(value):
    if isinstance(value, str) and value.endswith("s"):
        return int(value[:-1])  
    return int(value) if not pd.isnull(value) else None

# %%
# include subtitle_id (message_id), who (character), gender, cause (reason), social_emotion
all_bolly_df = all_bolly_df[["subtitle_id", "decade", "experience_social_emotion", "character", "social_emotion", "gender", "reason"]]
all_bolly_df['decade'] = all_bolly_df['decade'].apply(clean_decade).astype(int)
bolly_emotion_present = all_bolly_df[all_bolly_df["experience_social_emotion"] == "yes"]
bolly_shame = bolly_emotion_present[bolly_emotion_present["social_emotion"] == "shame"]
bolly_pride = bolly_emotion_present[bolly_emotion_present["social_emotion"] == "pride"]

print(f"num bolly_pride: {bolly_pride.shape[0]}")
print(f"num bolly_shame: {bolly_shame.shape[0]}")
bolly_shame.columns

# %%
# iclude subtitle_id (message_id), who (character), gender, cause (reason), social_emotion
all_holly_df = all_holly_df[["subtitle_id", "decade", "experience_social_emotion", "character", "social_emotion", "gender", "reason"]]
all_holly_df['decade'] = all_holly_df['decade'].apply(clean_decade).astype(int)
holly_emotion_present = all_holly_df[all_holly_df["experience_social_emotion"] == "yes"]
holly_shame = holly_emotion_present[holly_emotion_present["social_emotion"] == "shame"]
holly_pride = holly_emotion_present[holly_emotion_present["social_emotion"] == "pride"]

print(f"num holly_pride: {holly_pride.shape[0]}")
print(f"num holly_shame: {holly_shame.shape[0]}")

# %%
# drop NA reasons
bolly_shame = bolly_shame[bolly_shame['reason'].notna()]
holly_shame = holly_shame[holly_shame['reason'].notna()]

holly_pride = holly_pride[holly_pride['reason'].notna()]
bolly_pride = bolly_pride[bolly_pride['reason'].notna()]

# %%
print(f"num holly_pride: {holly_pride.shape[0]}")
print(f"num holly_shame: {holly_shame.shape[0]}")
print(f"num bolly_pride: {bolly_pride.shape[0]}")
print(f"num bolly_shame: {bolly_shame.shape[0]}")

# %%
### removing unknowns
searchfor = ['unknown', 'not specified']
bolly_shame = bolly_shame[~bolly_shame.reason.str.contains('|'.join(searchfor))]
holly_shame = holly_shame[~holly_shame.reason.str.contains('|'.join(searchfor))]
bolly_pride = bolly_pride[~bolly_pride.reason.str.contains('|'.join(searchfor))]
holly_pride = holly_pride[~holly_pride.reason.str.contains('|'.join(searchfor))]

print(f'after removing unknowns- bolly_shame={len(bolly_shame["reason"])}, holly_shame={len(holly_shame["reason"])}')
print(f'after removing unknowns- bolly_pride={len(bolly_pride["reason"])}, holly_pride={len(holly_pride["reason"])}')


# %%

## saving files
file_path = "processed_data/"
bolly_shame.to_csv(file_path+"bolly_shame_woNAs.csv")
holly_shame.to_csv(file_path+"holly_shame_woNAs.csv")
bolly_pride.to_csv(file_path+"bolly_pride_woNAs.csv")
holly_pride.to_csv(file_path+"holly_pride_woNAs.csv")

# %%
bolly_shame.head()

# %%
bolly_shame_decade = bolly_shame.groupby('decade').size()
# print(bolly_shame_decade)
bolly_pride_decade = bolly_pride.groupby('decade').size()
# print(bolly_pride_decade)
holly_shame_decade = holly_shame.groupby('decade').size()
print(holly_shame_decade)
holly_pride_decade = holly_pride.groupby('decade').size()
print(holly_pride_decade)

# %%
bolly_shame = pd.read_csv(folder_path+"bolly_shame_woNAs.csv")
bolly_shame.shape[0]

# %%
## WITHOUT DUPLICATES
## removing duplicates for gender analysis - there could be a particular genre over-present in one set of movies - skewing gender association - removing duplicates.
remove_duplicates = False


print(f'Not deduplicating - bolly_shame={len(bolly_shame["reason"])}, holly_shame={len(holly_shame["reason"])}')
print(f'Not deduplicating - bolly_pride={len(bolly_pride["reason"])}, holly_pride={len(holly_pride["reason"])}')
print("-"*50)

if remove_duplicates:
  bolly_shame = bolly_shame.drop_duplicates(subset='reason', keep="last")
  holly_shame = holly_shame.drop_duplicates(subset='reason', keep="last")
  bolly_pride = bolly_pride.drop_duplicates(subset='reason', keep="last")
  holly_pride = holly_pride.drop_duplicates(subset='reason', keep="last")

  print(f'after deduplicating - bolly_shame={len(bolly_pride["reason"])}, holly_shame={len(holly_shame["reason"])}')
  print(f'after deduplicating - bolly_pride={len(bolly_pride["reason"])}, holly_pride={len(holly_pride["reason"])}')

# %%
bolly_shame.head()

# %%
from collections import defaultdict

# %%
decade_list = [1960, 1970, 1980, 1990, 2000, 2010]
def create_df_list(df):
    df_list = defaultdict(list)
    for decade in decade_list:
        decade_df = df[(df['decade'] == decade)]
        df_list[decade] = decade_df
    return df_list

# %%
b_shame_list = create_df_list(bolly_shame)
b_pride_list = create_df_list(bolly_pride)

h_shame_list = create_df_list(holly_shame)
h_pride_list = create_df_list(holly_pride)


# %% [markdown]
# ### Clustering

# %%
# change it to include subtitle_id to allow for backtracking
bolly_shame = bolly_shame[["reason", "subtitle_id"]].values.tolist()
bolly_pride = bolly_pride[["reason", "subtitle_id"]].values.tolist()
holly_shame = holly_shame[["reason", "subtitle_id"]].values.tolist()
holly_pride = holly_pride[["reason", "subtitle_id"]].values.tolist()
all_shame = bolly_shame + holly_shame
all_pride = bolly_pride + holly_pride

# %%
len(all_shame)

# %%
len(bolly_shame)

# %%
len(holly_shame)

# %%
ENCODE = False # first time encoding
without_duplicate = True


# %%
folder_path = "processed_data/"

# %%
if(ENCODE == True and without_duplicate== True):
     #get embeddings from sbert for each sentence
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings_shame = model.encode(all_shame, show_progress_bar=True)
    pickle.dump(embeddings_shame, open(folder_path+"shame_embeddings_woduplicates.pkl", 'wb'))
    print("shame (without duplicates) embeddings saved")
    embeddings_pride = model.encode(all_pride, show_progress_bar=True)
    pickle.dump(embeddings_pride, open(folder_path+"pride_embeddings_woduplicates.pkl", 'wb'))
    print("pride (without duplicates) embeddings saved")

if ENCODE== True and without_duplicate==False:
    #get embeddings from sbert for each sentence
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings_shame = model.encode(all_shame, show_progress_bar=True)
    pickle.dump(embeddings_shame, open(folder_path+"shame_embeddings.pkl", 'wb'))
    print("shame embeddings (with duplicates) saved")
    embeddings_pride = model.encode(all_pride, show_progress_bar=True)
    pickle.dump(embeddings_pride, open(folder_path+"pride_embeddings.pkl", 'wb'))
    print("pride embeddings (with duplicates) saved")

if ENCODE== False and without_duplicate==True:
    embeddings_shame = pickle.load(open(folder_path+"shame_embeddings_woduplicates.pkl", 'rb'))
    embeddings_pride = pickle.load(open(folder_path+"pride_embeddings_woduplicates.pkl", 'rb'))
    print("embeddings without duplicates are loaded")

if ENCODE==False and without_duplicate == False:
    embeddings_shame = pickle.load(open(folder_path+"shame_embeddings.pkl", 'rb'))
    embeddings_pride = pickle.load(open(folder_path+"pride_embeddings.pkl", 'rb'))
    print("embeddings with duplicates are loaded.")
    


# %%
# all_shame

# %%

def create_embeddings(ENCODE, without_duplicate, all_shame, all_pride):

  folder_path = f"processed_data/{decade}"

  if(ENCODE == True and without_duplicate== True):
      #get embeddings from sbert for each sentence
      model = SentenceTransformer('all-mpnet-base-v2')
      embeddings_shame = model.encode(all_shame, show_progress_bar=True)
      pickle.dump(embeddings_shame, open(folder_path+"shame_embeddings_woduplicates.pkl", 'wb'))
      print("shame (without duplicates) embeddings saved")
      embeddings_pride = model.encode(all_pride, show_progress_bar=True)
      pickle.dump(embeddings_pride, open(folder_path+"pride_embeddings_woduplicates.pkl", 'wb'))
      print("pride (without duplicates) embeddings saved")

  if ENCODE== True and without_duplicate==False:
    #get embeddings from sbert for each sentence
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings_shame = model.encode(all_shame, show_progress_bar=True)
    pickle.dump(embeddings_shame, open(folder_path+"shame_embeddings.pkl", 'wb'))
    print("shame embeddings (with duplicates) saved")
    embeddings_pride = model.encode(all_pride, show_progress_bar=True)
    pickle.dump(embeddings_pride, open(folder_path+"pride_embeddings.pkl", 'wb'))
    print("pride embeddings (with duplicates) saved")

  if ENCODE== False and without_duplicate==True:
      embeddings_shame = pickle.load(open(folder_path+"shame_embeddings_woduplicates.pkl", 'rb'))
      embeddings_pride = pickle.load(open(folder_path+"pride_embeddings_woduplicates.pkl", 'rb'))
      print("embeddings without duplicates are loaded")

  if ENCODE==False and without_duplicate == False:
    embeddings_shame = pickle.load(open(folder_path+"shame_embeddings.pkl", 'rb'))
    embeddings_pride = pickle.load(open(folder_path+"pride_embeddings.pkl", 'rb'))
    print("embeddings with duplicates are loaded.")
    
  return embeddings_shame, embeddings_pride

# %%
#perform kmeans clustering on embeddings 
## distance_threshold = 5
## SET EMBEDDINGS
def set_embeddings(distance_threshold, emotion):

    if emotion == "shame":
        embeddings = embeddings_shame
        input = all_shame
        # embeddings = emb_shame_list[decade]
        # input = all_shame_list[decade]
    elif emotion == "pride":
        embeddings = embeddings_pride
        input = all_pride
        # embeddings = emb_pride_list[decade]
        # input = all_pride_list[decade]

    clustering_model = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=None)
    clustering_model.fit_predict(embeddings)
    cluster_assignment = clustering_model.labels_
    
    print(cluster_assignment)

    num_clusters = cluster_assignment.max()+1
    print("Number of clusters: ", num_clusters)

    ## PRIDE ##
    ## merging cluster assignment with gender-norm table
    cluster_assignment_df = pd.DataFrame(cluster_assignment, columns =['Cluster_ID'])
    #cluster_assignmentdf.head()

    input_df = pd.DataFrame(input, columns=['reason', 'subtitle_id']) 
    clustered_data = pd.concat([input_df, cluster_assignment_df], axis=1)
    # grouped_by_decade_reason = clustered_data.groupby(['decade', 'reason', 'Cluster_ID'])

    #save the file - cluster assignements
    file_path = f"processed_data/"
    clustered_data.to_csv(file_path+emotion+"_norms_dist"+str(distance_threshold)+"_decade_woduplicates.csv")
    
    return embeddings, num_clusters, cluster_assignment, input

# %% [markdown]
# ### Cross-validation for number of clusters 2,3,5,7,10

# %%
from sklearn.metrics import silhouette_score

def evaluate_clusters(embeddings, cluster_assignment):
    # Silhouette score is only valid if more than 1 cluster exists
    if len(set(cluster_assignment)) > 1:
        return silhouette_score(embeddings, cluster_assignment)
    else:
        return -1  # Return a bad score for single cluster


# %%
shame_results = defaultdict(list)
distance_thresholds = [2, 3, 4, 5, 7, 10]  

print("for shame:")
for threshold in distance_thresholds:
    embeddings, num_clusters, cluster_assignment, input = set_embeddings(threshold, emotion="shame")
    score = evaluate_clusters(embeddings, cluster_assignment)
    shame_results[threshold] = [score, embeddings, num_clusters, cluster_assignment, input]
    print(f"Threshold: {threshold}, Clusters: {num_clusters}, Score: {score}")

# %%
# shame_results[5]
# shame_results[5][4]

# %%
pride_results = defaultdict(list)
distance_thresholds = [2, 3, 4, 5, 7, 10]  

print("for pride:")
for threshold in distance_thresholds:
    embeddings, num_clusters, cluster_assignment, input = set_embeddings(threshold, emotion="pride")
    score = evaluate_clusters(embeddings, cluster_assignment)
    pride_results[threshold] = [score, embeddings, num_clusters, cluster_assignment, input]
    print(f"Threshold: {threshold}, Clusters: {num_clusters}, Score: {score}")

# %%
#let's take threshold = 5
threshold = 7
s_score, shame_embeddings, num_shame_clusters, shame_ca, shame_input = shame_results[threshold]
p_score, pride_embeddings, num_pride_clusters, pride_ca, pride_input = pride_results[threshold]

# %%
# testing out lower scores
threshold = 4
s_score, shame_embeddings, num_shame_clusters, shame_ca, shame_input = shame_results[threshold]
p_score, pride_embeddings, num_pride_clusters, pride_ca, pride_input = pride_results[threshold]

# %%
# pride_input

# %%
#get the 10 sentences closest to each cluster centers
def get_closest_sentences(embeddings, cluster_assignment, emotion, num_clusters, input_sentences):
    center_embeddings = []
    for i in range(num_clusters):
        cluster_center = embeddings[cluster_assignment==i].mean(axis=0)
        center_embeddings.append(cluster_center)
        
    print(f"center_embeddings: {center_embeddings}")

    closest_sentences = []
    for i in range(num_clusters):
        cluster_center = center_embeddings[i]
        distances = cosine_similarity([cluster_center], embeddings)
        closest = np.argsort(distances[0])[::-1][:10]
        ### change file name: all_shame or all_pride 
        # add subtitle_id too
        sentences = [(input_sentences[idx][0], input_sentences[idx][1]) for idx in closest]
        # sentences = [input_sentences[idx] for idx in closest]
        closest_sentences.append((i,sentences))
    
    print(f"center_embeddings: {closest_sentences}")


    # cluster_examples = pd.DataFrame(closest_sentences)
    cluster_examples = pd.DataFrame({
        "Cluster_ID": [cluster[0] for cluster in closest_sentences],
        "Closest_Sentences": [[(reason) for reason, _ in cluster[1]] for cluster in closest_sentences],
        "Subtitle_ID": [[(subtitle_id) for _, subtitle_id in cluster[1]] for cluster in closest_sentences],
    })

    ## saving files
    file_path = "processed_data/"
    # change threshold here
    cluster_examples.to_csv(file_path+emotion+"_id_norms_dist_"+str(threshold)+"_woduplicates_examples.csv")

    # for i, sentences in enumerate(closest_sentences):
    #     print("Cluster ", i, ": ", sentences)
    for cluster_id, sentences in closest_sentences:
        print(f"Cluster {cluster_id}:")
        for reason, subtitle_id in sentences:
            print(f"  Subtitle_ID: {subtitle_id}, Reason: {reason}")
        
    return cluster_examples

# %%
shame_examples = get_closest_sentences(shame_embeddings, shame_ca, "shame", num_shame_clusters, shame_input)
pride_examples = get_closest_sentences(pride_embeddings, pride_ca, "pride", num_pride_clusters, pride_input)

# %%
# shame_examples # top 20 matching sentences

# %%
pride_examples

# %%
filepath = "processed_data/"
bolly_shame = pd.read_csv(filepath+"bolly_shame_woNAs.csv")
holly_shame = pd.read_csv(filepath+"holly_shame_woNAs.csv")
bolly_pride = pd.read_csv(filepath+"bolly_pride_woNAs.csv")
holly_pride = pd.read_csv(filepath+"holly_pride_woNAs.csv")

## assign movie industry flag before concatenating tables.
bolly_shame['movie_industry']=1
holly_shame['movie_industry']=0
bolly_pride['movie_industry']=1
holly_pride['movie_industry']=0

## concatenation
all_shame = pd.concat([holly_shame, bolly_shame])
all_pride= pd.concat([bolly_pride, holly_pride])

# %%
all_shame.head()
# holly is 0, bolly is 1

# %%
pride_clusterIDs = pd.read_csv(filepath+"pride_norms_dist4_decade_woduplicates.csv")
shame_clusterIDs = pd.read_csv(filepath+"shame_norms_dist4_decade_woduplicates.csv")

# %%
pride_clusterIDs.head()

# %%
all_shame

# %%
shame_clusterIDs_decade = pd.merge(
    all_shame[['subtitle_id', 'reason', 'decade', 'gender', 'experience_social_emotion', 'movie_industry']], #1 is bolly, 0 is holly
    shame_clusterIDs[['reason', 'subtitle_id', 'Cluster_ID']],
    on=['subtitle_id', 'reason'],
    how='inner'
)
print(shame_clusterIDs_decade.shape[0])
shame_clusterIDs_decade.head()

# %%
pride_clusterIDs_decade = pd.merge(
    all_pride[['subtitle_id', 'reason', 'decade', 'movie_industry', 'gender']], #1 is bolly, 0 is holly
    pride_clusterIDs[['reason', 'subtitle_id', 'Cluster_ID']],
    on=['subtitle_id', 'reason'],
    how='inner'
)
print(pride_clusterIDs_decade.shape[0])
pride_clusterIDs_decade.head()

# %%
# map cluster ids to concatenated files
mapped_shame = pd.merge(all_shame, shame_clusterIDs, how="left", on="reason")
mapped_pride = pd.merge(all_pride, pride_clusterIDs, how="left", on="reason")

print("Original tables:", len(all_shame), len(all_pride))
print ("After mapping:", len(mapped_shame), len(mapped_pride))

# %%
file_path = "processed_data/"
shame_clusterIDs_decade.to_csv(filepath+"shameNormsMappedtoClusters.csv")
pride_clusterIDs_decade.to_csv(filepath+"prideNormsMappedtoClusters.csv")

# %%
pride_clusterIDs_decade = pd.read_csv(file_path+"prideNormsMappedtoClusters.csv")

# %%
pride_clusterIDs_decade.columns

# %%
shame_clusterIDs_decade.columns

# %%
pride_clusterIDs_decade.columns
Pride_clusterDistxmovieIndustry =  pride_clusterIDs_decade.groupby(['movie_industry', 'Cluster_ID', 'decade']).size().unstack(fill_value=0)

# %%
# Create the contingency table
Shame_clusterDistxmovieIndustry =  shame_clusterIDs_decade.groupby(['movie_industry', 'Cluster_ID', 'decade']).size().unstack(fill_value=0)
Pride_clusterDistxmovieIndustry =  pride_clusterIDs_decade.groupby(['movie_industry', 'Cluster_ID', 'decade']).size().unstack(fill_value=0)

# %%
Shame_clusterDistxmovieIndustry.to_csv(folder_path+"shame_cluster_decade.csv", index=True)

# %%
Pride_clusterDistxmovieIndustry

# %%
Shame_clusterDistxmovieIndustry

# %%
Shame_clusterDistxmovieIndustry.T.to_csv(folder_path+"shame_cluster_decade.csv", index=True)

# %%
Pride_clusterDistxmovieIndustry.to_csv(file_path+"pride_cluster_decade.csv", index=True)

# %%
print("Bolly-shame:", bolly_shame['gender'].value_counts())
print("Holly-shame:", holly_shame['gender'].value_counts())
print("Bolly-pride:", bolly_pride['gender'].value_counts())
print("Holly-pride:", holly_pride['gender'].value_counts())

# %%
mapped_shame_gender = shame_clusterIDs_decade.loc[(shame_clusterIDs_decade['gender'] == "male") | (shame_clusterIDs_decade['gender'] == "female")]
mapped_pride_gender = pride_clusterIDs_decade.loc[(pride_clusterIDs_decade['gender'] == "male") | (pride_clusterIDs_decade['gender'] == "female")]

# %%
print(f"pride: {mapped_pride_gender['gender'].value_counts()}")
print(f"{mapped_shame_gender['gender'].value_counts()}")

# %%
gender_clusterdist_shame = mapped_shame_gender.groupby([ 'movie_industry', 'Cluster_ID','gender']).size().unstack(fill_value=0)
gender_clusterdist_pride = mapped_pride_gender.groupby([ 'movie_industry', 'Cluster_ID','gender']).size().unstack(fill_value=0)

# %%
gender_clusterdist_shame = gender_clusterdist_shame.reset_index()
gender_clusterdist_pride = gender_clusterdist_pride.reset_index()

# %%
genderxnorms_shame_bolly = gender_clusterdist_shame[gender_clusterdist_shame["movie_industry"]==1]
genderxnorms_shame_holly = gender_clusterdist_shame[gender_clusterdist_shame["movie_industry"]==0]

genderxnorms_pride_bolly = gender_clusterdist_pride[gender_clusterdist_pride["movie_industry"]==1]
genderxnorms_pride_holly = gender_clusterdist_pride[gender_clusterdist_pride["movie_industry"]==0]

# %%
genderxnorms_shame_bolly = genderxnorms_shame_bolly.reset_index(drop=True)
genderxnorms_shame_holly = genderxnorms_shame_holly.reset_index(drop=True)

genderxnorms_pride_bolly = genderxnorms_pride_bolly.reset_index(drop=True)
genderxnorms_pride_holly = genderxnorms_pride_holly.reset_index(drop=True)

# %%
genderxnorms_pride_bolly.columns

# %% [markdown]
# ### Visualization

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
file_path = "processed_data/"
pride_cluster_df = pd.read_csv(file_path+"pride_cluster_df.csv")
shame_cluster_df = pd.read_csv(file_path+"shame_cluster_df.csv")

# %%
pride_cluster_df.columns = pride_cluster_df.iloc[0]
pride_cluster_df = pride_cluster_df.drop(0).reset_index(drop=True)
pride_cluster_df

# %%
pride_cluster_df['Cluster_ID'] = pride_cluster_df.index  # Index will give you 0 to 14, so no need to manually define this.
pride_cluster_df.head()

# %%
pride_labels = [
    "Proud of Someone Else's Achievements",
    "Superior Recognition",
    "Public Praise and Appreciation",
    "Proud of Family Member's Achievements",
    "Making Family Members Proud",
    "Fulfilling Familial Roles",
    "Bravery",
    "Community Pride",
    "Career Success",
    "Confronting Adversity",
    "Award Recognition",
    "Family Recognition",
    "Parental Pride",
    "Beauty",
    "Public Recognition",
    "Self-Respect"
]

pride_id_to_label = {idx: label for idx, label in enumerate(pride_labels)}

# Print the mapping
print(pride_id_to_label)


# %%
# pride_id_to_label

# %%
shame_labels = [
    "Criminal Misconduct",
    "Public Humiliation",
    "Public Criticism",
    "External Perception and Shame",
    "Internalized Shame",
    "Dishonorable Familial Relationships",
    "Male Disappointing Family",
    "Parental Shame",
    "Betrayal",
    "Misconducts and Accusations",
    "Cowardrice",
    "Female Disappointing Family",
    "Theft",
    "Internalized Shame from Family Members",
    "Irresponsibility"
]
shame_id_to_label = {idx: label for idx, label in enumerate(shame_labels)}
print(shame_id_to_label)

# %%
pride_cluster_df['c_label'] = pride_cluster_df['Cluster_ID'].map(pride_id_to_label)

# %%
pride_cluster_df.head()

# %%
pride_cluster_df.T

# %% [markdown]
# ### Creating the graphs: PRIDE

# %%
# run code block here to clean df for graphing
final_df = pride_cluster_df.T
final_df.columns = final_df.iloc[0]
final_df.head()


# %%
final_df.drop(final_df.index[:1], inplace=True)

# %%
# final_df.rename(columns={'Cluster_ID': 'decade'}, inplace=True)
final_df.head()


# %%
# pride_cluster_df = pride_cluster_df.drop('c_label', axis=1)

# %%
# pride_cluster_df

# %%
pride_labels = [
    "Superior Recognition",
    "Public Praise and Appreciation",
    "Career Success",
    "Confronting Adversity",
    "Award Recognition"
]

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# PRIDE: Run this code block to show ALL the PRIDE clusters

# Melt the DataFrame to a long format
df_melted = pride_cluster_df.melt(id_vars=["Cluster_ID", "c_label"], var_name="Decade", value_name="Delta")
df_melted["Decade"] = df_melted["Decade"].astype(int)  

clusters = df_melted["Cluster_ID"].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(clusters)))  
cluster_colors = dict(zip(clusters, colors))

plt.figure(figsize=(12, 8))

# Plot a line for each cluster
for cluster_id in clusters:
    subset = df_melted[df_melted["Cluster_ID"] == cluster_id]
    plt.plot(
        subset["Decade"],
        subset["Delta"],
        label=f"Cluster {cluster_id}: {subset['c_label'].iloc[0]}",
        color=cluster_colors[cluster_id],
        marker="o",
    )

plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.xticks(
    ticks=sorted(df_melted["Decade"].unique()), 
    labels=sorted(df_melted["Decade"].unique()),
)
plt.xlabel("Decade")
plt.ylabel("Relative Association (Δ)")
plt.title("Relative Association of Pride Themes Across Decades")
plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

plt.show()


# %%
# TODO: edit cluster_map to onlu include the clusters you want

cluster_map = ["Misconducts", "Family Recognition", "Community Pride"]  # Add your desired cluster labels here

# Filter the DataFrame to include only the specified clusters
df_filtered = df_melted[df_melted["c_label"].isin(pride_labels)]

# Generate unique colors for each cluster in the filtered DataFrame
clusters = df_filtered["Cluster_ID"].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(clusters)))  # Adjust colormap as needed
cluster_colors = dict(zip(clusters, colors))

plt.figure(figsize=(12, 8))

# Plot a line for each filtered cluster
for cluster_id in clusters:
    subset = df_filtered[df_filtered["Cluster_ID"] == cluster_id]
    plt.plot(
        subset["Decade"],
        subset["Delta"],
        label=f"Cluster {cluster_id}: {subset['c_label'].iloc[0]}",
        color=cluster_colors[cluster_id],
        marker="o",
    )

# Customize the plot
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.xticks(
    ticks=sorted(df_filtered["Decade"].unique()),  # Ensure all decades are shown
    labels=sorted(df_filtered["Decade"].unique()),
)
plt.xlabel("Decade")
plt.ylabel("Relative Association (Δ)")
plt.title("Relative Association of Selected Shame Themes Across Decades")
plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

# Show the plot
plt.show()


# %% [markdown]
# ### Creating the graphs: SHAME

# %%
shame_cluster_df = pd.read_csv(file_path+"shame_cluster_df.csv")

# %%
shame_cluster_df.columns = shame_cluster_df.iloc[0]
shame_cluster_df = shame_cluster_df.drop(0).reset_index(drop=True)
shame_cluster_df.head()

# %%
shame_cluster_df['Cluster_ID'] = shame_cluster_df.index  # Index will give you 0 to 14, so no need to manually define this.
shame_cluster_df.head()

# %%
shame_cluster_df['c_label'] = shame_cluster_df['Cluster_ID'].map(shame_id_to_label)

# %%
# TODO: edit this to contain only the clusters you want to show
bolly_shame = [
    "Public Humiliation",
    "Public Criticism",
    "External Perception and Shame",
    "Internalized Shame"
]


# %%

df_melted = shame_cluster_df.melt(id_vars=["Cluster_ID", "c_label"], var_name="Decade", value_name="Delta")
df_melted["Decade"] = df_melted["Decade"].astype(int)  

clusters = df_melted["Cluster_ID"].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(clusters))) 
cluster_colors = dict(zip(clusters, colors))

plt.figure(figsize=(12, 8))

for cluster_id in clusters:
    subset = df_melted[df_melted["Cluster_ID"] == cluster_id]
    plt.plot(
        subset["Decade"],
        subset["Delta"],
        label=f"Cluster {cluster_id}: {subset['c_label'].iloc[0]}",
        color=cluster_colors[cluster_id],
        marker="o",
    )

plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.xticks(
    ticks=sorted(df_melted["Decade"].unique()),  
    labels=sorted(df_melted["Decade"].unique()),
)
plt.xlabel("Decade")
plt.ylabel("Relative Association (Δ)")
plt.title("Relative Association of Shame Themes Across Decades")
plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

# Show the plot
plt.show()

# %%

# Filter the DataFrame to include only the specified clusters
df_filtered = df_melted[df_melted["c_label"].isin(bolly_shame)]

# Generate unique colors for each cluster in the filtered DataFrame
clusters = df_filtered["Cluster_ID"].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(clusters)))  # Adjust colormap as needed
cluster_colors = dict(zip(clusters, colors))

plt.figure(figsize=(12, 8))

# Plot a line for each filtered cluster
for cluster_id in clusters:
    subset = df_filtered[df_filtered["Cluster_ID"] == cluster_id]
    plt.plot(
        subset["Decade"],
        subset["Delta"],
        label=f"Cluster {cluster_id}: {subset['c_label'].iloc[0]}",
        color=cluster_colors[cluster_id],
        marker="o",
    )

# Customize the plot
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.xticks(
    ticks=sorted(df_filtered["Decade"].unique()),  # Ensure all decades are shown
    labels=sorted(df_filtered["Decade"].unique()),
)
plt.xlabel("Decade")
plt.ylabel("Relative Association (Δ)")
plt.title("Relative Association of Selected Shame Themes Across Decades")
plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

# Show the plot
plt.show()


