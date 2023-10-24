# crop = 'broad_bean'
crop = 'common_buckwheat'
# crop = 'pea'
# crop = 'corn'
# crop = 'soybean'
# crop = 'sunflower'
# crop = 'sugar_beet'

# model_type = 'multiclass'
model_type = 'binary'

crops = [ 
    'broad_bean',
    'common_buckwheat',
    'pea',
    'corn',
    'soybean',
    'sunflower',
    'sugar_beet'
]

model_types = ['binary', 'multiclass']


# checkpoint = "nvidia/mit-b0"
# checkpoint = "facebook/maskformer-swin-large-coco"
# checkpoint = "microsoft/beit-base-finetuned-ade-640-640"
# checkpoint = "Intel/dpt-large-ade"
# checkpoint = "apple/deeplabv3-mobilevit-xx-small"
checkpoint = "openmmlab/upernet-swin-tiny"


batch_size = 8
# batch_size = 2