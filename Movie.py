import pandas as pd
import ast
from tqdm import tqdm
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import json
# Load sentence transformer model
model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

# Load the datasets
movies_df = pd.read_csv('tmdb_5000_movies.csv')
credits_df = pd.read_csv('tmdb_5000_credits.csv')

merged_df = pd.merge(movies_df, credits_df, left_on='id', right_on='movie_id')

keywords_df = pd.read_csv('keywords.csv')
links_df = pd.read_csv('links.csv')
movies_metadata_df = pd.read_csv('movies_metadata.csv', low_memory=False)


# # Display the first few rows of each new dataframe to understand their structure
# keywords_df.shape, links_df.shape, movies_metadata_df.shape

# Merge the keywords_df with the previously merged dataset
merged_with_keywords_df = pd.merge(merged_df, keywords_df, on='id', how='left')

# Now merge the movies_metadata_df with the result, aligning on 'id'
# We need to ensure the 'id' columns are of the same type
merged_with_keywords_df['id'] = merged_with_keywords_df['id'].astype(str)
movies_metadata_df['id'] = movies_metadata_df['id'].astype(str)

final_merged_df = pd.merge(merged_with_keywords_df, movies_metadata_df, on='id', how='left', suffixes=('_left', '_right'))

# # Display the shape of the final merged dataframe to understand the extent of merging
# final_merged_df.shape, final_merged_df.columns

# Identify columns with '_left' and '_right' suffixes and prepare for comparison and potential consolidation
duplicate_columns = [col for col in final_merged_df.columns if '_left' in col or '_right' in col]
unique_columns = [col.replace('_left', '').replace('_right', '') for col in duplicate_columns]

# Create a dictionary to map unique column names to their '_left' and '_right' counterparts
column_mapping = {}
for unique_col in set(unique_columns):
    left_version = f"{unique_col}_left"
    right_version = f"{unique_col}_right"
    column_mapping[unique_col] = [left_version, right_version]

# Determine strategy for each pair: either keep one (if identical or one is preferable) or merge (if complementary)
# This will be a manual process based on column content. For simplicity, we'll initially assume to keep '_left' versions
# and drop '_right', unless inspection reveals a need for a different approach.

# Columns to keep without change (initially all '_left' versions and unique ones without such suffixes)
columns_to_keep = [col for col in final_merged_df.columns if '_right' not in col]

# Drop '_right' columns from the dataframe
cleaned_df = final_merged_df[columns_to_keep].copy()

# For demonstration, let's rename '_left' suffixes to remove them and clean up column names
cleaned_df.columns = [col.replace('_left', '') if '_left' in col else col for col in cleaned_df.columns]

# Display the cleaned dataframe structure
# cleaned_df.head(), cleaned_df.columns.tolist()

# cleaned_df.columns,cleaned_df.shape

remove_columns = ['keywords_y','title_y']
cleaned_df = cleaned_df.drop(remove_columns,axis=1)
# cleaned_df.shape
def extract_data(col, df):
    genres_list = []
    for index, row in tqdm(df.iterrows()):
        genres_info = row[col]

        # Use json.loads instead of ast.literal_eval
        try:
            genres_list_row = json.loads(genres_info)
            all_genres_types = [i.get('name') for i in genres_list_row]
            genres_str = ','.join([str(elem) for elem in all_genres_types])
            genres_list.append(genres_str)
        except json.JSONDecodeError:
            # Handle the case where the data is not valid JSON
            st.warning(f"Error decoding JSON in column '{col}', index {index}")

    return genres_list

cleaned_genres = extract_data('genres', cleaned_df)
cleaned_df['cleaned_genres'] = cleaned_genres

cleaned_keywords = extract_data('keywords_x', cleaned_df)
cleaned_df['cleaned_keywords'] = cleaned_keywords

cleaned_prod_comp = extract_data('production_companies', cleaned_df)
cleaned_df['cleaned_production_companies'] = cleaned_prod_comp

cleaned_prod_countries = extract_data('production_countries', cleaned_df)
cleaned_df['cleaned_production_countries'] = cleaned_prod_countries

cleaned_spoken_languages = extract_data('spoken_languages', cleaned_df)
cleaned_df['cleaned_spoken_languages'] = cleaned_spoken_languages

get_character_name = extract_data('cast', cleaned_df)
cleaned_df['cleaned_cast'] = get_character_name

get_crew_name = extract_data('crew', cleaned_df)
cleaned_df['cleaned_crew'] = get_crew_name

cleaned_df['combined_text'] = cleaned_df['cleaned_genres'] + ' ' + cleaned_df['cleaned_keywords'] + ' ' + cleaned_df['original_language']+ ' ' + cleaned_df['overview'] + ' ' + cleaned_df['cleaned_production_companies'] + ' ' +  cleaned_df['cleaned_production_countries'] + ' ' + cleaned_df['cleaned_spoken_languages'] + ' ' + cleaned_df['tagline'] + ' ' + cleaned_df['cleaned_cast'] + ' ' + cleaned_df['cleaned_crew']


# Load preprocessed dataframe with embeddings
unpickled_df = pd.read_pickle("embeddings_df_bert.pkl")

# # Load sentence transformer model
# model = SentenceTransformer('your_model_name_or_path')

# Function to get similar items based on combined name and category
def get_similar_items(combined_input, df, model, top_n=6):
    top_n = top_n + 1
    combined_embedding = model.encode([combined_input])
    similarities = util.pytorch_cos_sim(combined_embedding, df['combined_embeddings'])
    
    # Convert the similarities tensor to a numpy array
    similarities_np = similarities.cpu().numpy()

    # Use numpy argsort to get indices of sorted similarities
    similar_indices = (-similarities_np).argsort(axis=1)[0][:top_n]

    # Get similar items based on the sorted indices
    similar_items_df = df.iloc[similar_indices][['title']]
    return similar_items_df





# Inference is done

import warnings
warnings.filterwarnings('ignore')
infer_df = cleaned_df.head(1)

infer_cleaned_genres = extract_data('genres', infer_df)
infer_df['cleaned_genres'] = infer_cleaned_genres

infer_cleaned_keywords = extract_data('keywords_x', infer_df)
infer_df['cleaned_keywords'] = infer_cleaned_keywords

infer_cleaned_prod_comp = extract_data('production_companies', infer_df)
infer_df['cleaned_production_companies'] = infer_cleaned_prod_comp

infer_prod_countries = extract_data('production_countries', infer_df)
infer_df['cleaned_production_countries'] = infer_prod_countries

infer_spoken_languages = extract_data('spoken_languages', infer_df)
infer_df['cleaned_spoken_languages'] = infer_spoken_languages

infer_get_character_name = extract_data('cast', infer_df)
infer_df['cleaned_cast'] = infer_get_character_name

infer_get_crew_name = extract_data('crew', infer_df)
infer_df['cleaned_crew'] = infer_get_crew_name

infer_df['combined_text'] = infer_df['cleaned_genres'] + ' ' + infer_df['cleaned_keywords'] + ' ' + infer_df['original_language']+ ' ' + infer_df['overview'] + ' ' + infer_df['cleaned_production_companies'] + ' ' +  infer_df['cleaned_production_countries'] + ' ' + infer_df['cleaned_spoken_languages'] + ' ' + infer_df['tagline'] + ' ' + infer_df['cleaned_cast'] + ' ' + infer_df['cleaned_crew']

infer_df['combined_text_str'] = infer_df['combined_text'].apply(lambda x: str(x))
# Streamlit app
st.title('Movie Similarity Search App')

selected_movie = st.selectbox('Select a Movie:', cleaned_df['title'])
top_k=st.number_input("select the number of recommended movies", step=1, format="%d")
# Get similar items based on user-selected movie
if st.button('Find Similar Movies'):
    product_info_to_search = cleaned_df[cleaned_df['title'] == selected_movie]['combined_text'].values[0]
    similar_items = get_similar_items(product_info_to_search, unpickled_df, model,top_n=top_k)

    # Exclude the selected movie from the list
    similar_items = similar_items[similar_items['title'] != selected_movie]

    # Print similar movies
    st.subheader('Similar Movies:')
    for idx, row in similar_items.iterrows():
        st.write(row['title'])




