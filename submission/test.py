import json
data = json.load(open("answer.json", encoding="utf8"))
for i in data:
    if i['answer'] == "Khong biet":
        i['answer'] = None
 
obj = {
    "data": data
}   
json.dump(obj, open("submission.json", "w", encoding="utf8"), ensure_ascii=False, indent=4)