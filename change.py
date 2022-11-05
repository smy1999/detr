import torch
# pretrained_weights = torch.load('detr-r50-e632da11.pth')
#
# num_class = 2
# num_queries = 200
# pretrained_weights['model']['class_embed.weight'].resize_(num_class + 1, 256)
# pretrained_weights['model']['class_embed.bias'].resize_(num_class + 1)
# pretrained_weights['model']['query_embed.weight'].resize_(num_queries, 256)
# torch.save(pretrained_weights, 'detr-r50_%d.pth' % num_class)

# panoptic
# pretrained_weights = torch.load('detr-r50-panoptic-00ce5173.pth')
#
# num_class = 2
# num_queries = 666
# pretrained_weights['model']['detr.class_embed.weight'].resize_(num_class + 1, 256)
# pretrained_weights['model']['detr.class_embed.bias'].resize_(num_class + 1)
# pretrained_weights['model']['detr.query_embed.weight'].resize_(num_queries, 256)
# torch.save(pretrained_weights, 'detr-r50_p%d.pth' % num_class)

# remove classification head
pretrained_weights = torch.load('detr-r50-panoptic-00ce5173.pth')

num_class = 2
num_queries = 100
del pretrained_weights['model']['detr.class_embed.weight']
del pretrained_weights['model']['detr.class_embed.bias']
# pretrained_weights['model']['detr.query_embed.weight'].resize_(num_queries, 256)
torch.save(pretrained_weights, 'detr-r50_p%d.pth' % num_class)



def change_key(num_class):
    pretrained_weights_panoptic = torch.load('detr-r50-panoptic-00ce5173.pth')
    d = pretrained_weights_panoptic['model'].keys()
    for old_key in list(d):
        if old_key[:5] == 'detr.':
            new_key = old_key[5:]
            pretrained_weights_panoptic['model'][new_key] = pretrained_weights_panoptic['model'][old_key]
            del pretrained_weights_panoptic['model'][old_key]
        else:
            del pretrained_weights_panoptic['model'][old_key]
    torch.save(pretrained_weights_panoptic, 'detr-r50_p%d.pth' % num_class)

    # pretrained_weights['model']['detr.class_embed.weight'].resize_(num_class + 1, 256)
    # pretrained_weights['model']['detr.class_embed.bias'].resize_(num_class + 1)
    # pretrained_weights['model']['detr.query_embed.weight'].resize_(num_queries, 256)
    # torch.save(pretrained_weights, 'detr-r50_p%d.pth' % num_class)


# if __name__ == '__main__':
#     num_class = 2
#     change_key(2)
