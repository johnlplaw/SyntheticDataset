from sentence_transformers import SentenceTransformer, util
import torch
import ValidationFunction as commf
import ValidationObject as commO
from bert_score import score

# Define the
table_str = "similarity_st_XLMR"

# Move model to GPU if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = SentenceTransformer('FacebookAI/xlm-roberta-base', device=device)

def execution():

    # Step 1: load all dataset
    oritxt_list = commf.get_Syn_Dataset(table_str, "LIMIT 0, 100")
    # Step 2: Similarity


    for txtObj in oritxt_list:
        oritxt1_list = []
        txt2_list = []
        for field in commO.field_list:
            oritxt1_list.append(txtObj.cleanedtxt)
            txt2_list.append(txtObj.get_txt(field))

        embedding_1 = model.encode(oritxt1_list, convert_to_tensor=True, device=device)
        embedding_2 = model.encode(txt2_list, convert_to_tensor=True, device=device)

        # Compute cosine similarity
        similarity = util.cos_sim(embedding_1, embedding_2)

        # Output
        for i in range(len(commO.field_list)):
            commf.insertUpdateSimilarityType1(table_str, txtObj.id, commO.field_list[i], round(similarity[i][i].item(), 4))

#execution()

count = 0;
while count < 500:
    execution()
    count = count + 1
