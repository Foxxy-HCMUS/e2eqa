from models.pairwise_model import *
from features.text_utils import *
import regex as re
from models.bm25_utils import BM25Gensim
from models.qa_model import *
from tqdm.auto import tqdm
tqdm.pandas()

df_wiki_windows = pd.read_csv("../data/processed/wikipedia_20220620_cleaned_v2.csv")
df_wiki = pd.read_csv("../data/wikipedia_20220620_short.csv")
df_wiki.title = df_wiki.title.apply(str)

entity_dict = json.load(open("../data/processed/entities.json"))
new_dict = dict()
for key, val in entity_dict.items():
    val = val.replace("wiki/", "").replace("_", " ")
    entity_dict[key] = val
    key = preprocess(key)
    new_dict[key.lower()] = val
entity_dict.update(new_dict)
title2idx = dict([(x.strip(), y) for x, y in zip(df_wiki.title, df_wiki.index.values)])

qa_model = QAEnsembleModel("nguyenvulebinh/vi-mrc-large", ["../models/qa_model_robust.bin"], entity_dict)
pairwise_model_stage1 = PairwiseModel("nguyenvulebinh/vi-mrc-base").half()
pairwise_model_stage1.load_state_dict(torch.load("../models/pairwise_v2.bin"))
pairwise_model_stage1.eval()

pairwise_model_stage2 = PairwiseModel("nguyenvulebinh/vi-mrc-base").half()
pairwise_model_stage2.load_state_dict(torch.load("../models/pairwise_stage2_seed0.bin"))

bm25_model_stage1 = BM25Gensim("../models/bm25_stage1/", entity_dict, title2idx)
bm25_model_stage2_full = BM25Gensim("../models/bm25_stage2/full_text/", entity_dict, title2idx)
bm25_model_stage2_title = BM25Gensim("../models/bm25_stage2/title/", entity_dict, title2idx)

def get_answer_e2e(question):
    #Bm25 retrieval for top200 candidates
    query = preprocess(question).lower()
    top_n, bm25_scores = bm25_model_stage1.get_topk_stage1(query, topk=200)
    titles = [preprocess(df_wiki_windows.title.values[i]) for i in top_n]
    texts = [preprocess(df_wiki_windows.text.values[i]) for i in top_n]
    
    #Reranking with pairwise model for top10
    question = preprocess(question)
    ranking_preds = pairwise_model_stage1.stage1_ranking(question, texts)
    ranking_scores = ranking_preds * bm25_scores
    
    #Question answering
    best_idxs = np.argsort(ranking_scores)[-10:]
    ranking_scores = np.array(ranking_scores)[best_idxs]
    texts = np.array(texts)[best_idxs]
    best_answer = qa_model(question, texts, ranking_scores)
    if best_answer is None:
        return "Chịu"
    bm25_answer = preprocess(str(best_answer).lower(), max_length=128, remove_puncts=True)
    
    #Entity mapping
    if not check_number(bm25_answer):
        bm25_question = preprocess(str(question).lower(), max_length=128, remove_puncts=True)
        bm25_question_answer = bm25_question + " " + bm25_answer
        candidates, scores = bm25_model_stage2_title.get_topk_stage2(bm25_answer, raw_answer=best_answer)
        titles = [df_wiki.title.values[i] for i in candidates]
        texts = [df_wiki.text.values[i] for i in candidates]
        ranking_preds = pairwise_model_stage2.stage2_ranking(question, best_answer, titles, texts)
        if ranking_preds.max() >= 0.1:
            final_answer = titles[ranking_preds.argmax()]
        else:
            candidates, scores = bm25_model_stage2_full.get_topk_stage2(bm25_question_answer)
            titles = [df_wiki.title.values[i] for i in candidates] + titles
            texts = [df_wiki.text.values[i] for i in candidates] + texts
            ranking_preds = np.concatenate(
                [pairwise_model_stage2.stage2_ranking(question, best_answer, titles, texts), ranking_preds])
        final_answer = "wiki/"+titles[ranking_preds.argmax()].replace(" ","_")
    else:
        final_answer = bm25_answer.lower()
    return final_answer