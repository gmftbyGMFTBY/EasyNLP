import json

# source_file = open("../TGDR/TGConv/ours/easy.txt", "r")
# target_file = open("../TGDR/TGConv/ours/easy.jsonl", "w")
source_file = open("../TGDR/TGConv/ours/hard.txt", "r")
target_file = open("../TGDR/TGConv/ours/hard.jsonl", "w")
# source_file = open("../TGDR/TGCP/ours/result.txt", "r")
# target_file = open("../TGDR/TGCP/ours/result.jsonl", "w")

source = source_file.readlines()

keyword = ""
dialogs = []
for idx, line in enumerate(source):
    if "[Context]" in line:
        if idx != 0:
            json.dump({"target_word": keyword, "conversation_plan": "[SEP]".join(dialogs)}, target_file)
            target_file.write("\n")
        dialogs = []
        keyword = ""
        dialogs.append(line.strip()[10:])
    elif "[Topic]" in line:
        keyword = line.strip()[8:]
    elif "[Distance" in line:
        if "chatbot:" in line:
            dialogs.append(line.strip().split("chatbot: ")[1])
        elif "human:" in line:
            dialogs.append(line.strip().split("human: ")[1])

json.dump({"target_word": keyword, "conversation_plan": "[SEP]".join(dialogs)}, target_file)
target_file.write("\n")
