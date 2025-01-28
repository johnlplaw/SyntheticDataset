from sentence_transformers import SentenceTransformer, util

import ValidationFunction as commf
import ValidationObject as commO
from bert_score import score

table_str = "similarity_st"

# Step 1: load all dataset
oritxt_list = commf.get_Syn_Dataset(table_str, "LIMIT 0, 20000")

# Step 2: Prepare the model
# Load a multilingual model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Step 3: Similarity
for txtObj in oritxt_list:
    for field in commO.field_list:
        if txtObj.get_txt(field) is None:
            print("Id:" + str(txtObj.id) + ", field:" + str(txtObj.get_txt(field)))
        else:
            similarity = commf.getSimilarityBySentenceTransformer(model, txtObj.cleanedtxt, txtObj.get_txt(field))
            #print(str(similarity))
            commf.insertUpdateSimilarityType1("similarity_st", txtObj.id, field, similarity)


