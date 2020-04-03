import json


file_path = "/home/vishal/datasets/annotations_trainval2017/annotations/instances_val2017.json"

f = open(file_path,)

data = json.load(f)
print(data['annotations'][0].keys())
print(data['annotations'][0])

x = []
for node in data['annotations']:
    x.append(node['image_id'])
    #print(node['image_id'], node['bbox'], node['id'])

x= tuple(x)
print(max(x), len(x))

