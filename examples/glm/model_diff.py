import torch

pt1 = "/home/tsm/checkpoints/finetune-glm-12-27-23-56/20/mp_rank_00_model_states.pt"
# pt2 = "/home/tsm/checkpoints/finetune-glm-12-27-23-56/20/mp_rank_00_model_states.pt"
pt2 = "/home/tsm/.sat_models/glm-large-en-blank/250000/mp_rank_00_model_states.pt"

model1 = torch.load(pt1, map_location=torch.device("cuda"))
model2 = torch.load(pt2, map_location=torch.device("cuda"))

cnt = 1
print(model1["module"]["mixins.prefix-tuning.prefix.0"].size()) # one for each layer
print(model1["module"]["mixins.prefix-tuning.prefix.0"])
print(model2["module"]["mixins.prefix-tuning.prefix.0"]) #shouldn't exist
# following should both be frozen
print(model1["module"]["transformer.layers.0.attention.query_key_value.weight"]) 
print(model2["module"]["transformer.layers.0.attention.query_key_value.weight"]) 
# print(model1["module"].keys())
# for param in model1:
#     if cnt > 1:
#         break
#     print("Model1")
#     print(model1[param])
#     print("-"*50)
#     print("Model2")
#     print(model2[param])
#     cnt += 1