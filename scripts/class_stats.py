from pycocotools.coco import COCO

coco = COCO("datasets/coco/annotations/instances_train2017.json")
categories = ["person", "car", "cell phone", "laptop", "book", "handbag", "sports ball"]

for cat in categories:
    cat_id = coco.getCatIds(catNms=[cat])[0]
    img_ids = coco.getImgIds(catIds=[cat_id])
    print(f"{cat:12s}: {len(img_ids):5d} images")
    
"""
person      : 64115 images
car         : 12251 images
cell phone  :  4803 images
laptop      :  3524 images
book        :  5332 images
handbag     :  6841 images
sports ball :  4262 images

"""