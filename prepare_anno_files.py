split_list_path = 'data/list_eval_partition.txt'
train_ids = set()
val_ids = set()
test_ids = set()

with open(split_list_path) as f:
    lines = f.readlines()[2:]
    for line in lines:
        img_id, split = line.strip().split()
        if split == '0':
            train_ids.add(img_id)
        elif split == '1':
            val_ids.add(img_id)
        elif split == '2':
            test_ids.add(img_id)


attr_anno_path = 'data/list_attr_celeba.txt'
attr_anno_train = []
attr_anno_val = []
attr_anno_test = []

with open(attr_anno_path) as f:
    lines = f.readlines()[2:]
    for line in lines:
        line = line.replace('-1', '0')
        item = line.strip().split()
        if item[0] in train_ids:
            attr_anno_train.append(line)
        elif item[0] in val_ids:
            attr_anno_val.append(line)
        elif item[0] in test_ids:
            attr_anno_test.append(line)

attr_anno_train_path = 'data/train_40_att_list.txt'
attr_anno_val_path = 'data/val_40_att_list.txt'
attr_anno_test_path = 'data/test_40_att_list.txt'

with open(attr_anno_train_path, 'w') as f:
    f.writelines(attr_anno_train)
    print('Train: %d' % len(attr_anno_train))
with open(attr_anno_val_path, 'w') as f:
    f.writelines(attr_anno_val)
    print('Val: %d' % len(attr_anno_val))
with open(attr_anno_test_path, 'w') as f:
    f.writelines(attr_anno_test)
    print('Test: %d' % len(attr_anno_test))
