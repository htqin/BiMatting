"""
Expected directory format:

VideoMatte Train/Valid:
    ├──fgr/
      ├── 0001/
        ├── 00000.jpg
        ├── 00001.jpg
    ├── pha/
      ├── 0001/
        ├── 00000.jpg
        ├── 00001.jpg
        
ImageMatte Train/Valid:
    ├── fgr/
      ├── sample1.jpg
      ├── sample2.jpg
    ├── pha/
      ├── sample1.jpg
      ├── sample2.jpg

Background Image Train/Valid
    ├── sample1.png
    ├── sample2.png

Background Video Train/Valid
    ├── 0000/
      ├── 0000.jpg/
      ├── 0001.jpg/

"""


DATA_PATHS = {
    'videomatte_SD': {
        'train': '/cluster/work/cvl/haoqin/datasets/matting-data/VideoMatte240K_JPEG_SD/train',
        'valid': '/cluster/work/cvl/haoqin/datasets/matting-data/VideoMatte240K_JPEG_SD/valid',
    },
    'videomatte_HD': {
        'train': '/cluster/work/cvl/haoqin/datasets/matting-data/VideoMatte240K_JPEG_SD/train',
        'valid': '/cluster/work/cvl/haoqin/datasets/matting-data/VideoMatte240K_JPEG_SD/valid',
    },
    'imagematte': {
        'train': '/cluster/work/cvl/haoqin/datasets/matting-data/ImageMatte/train',
        'valid': '/cluster/work/cvl/haoqin/datasets/matting-data/ImageMatte/valid',
    },
    'background_images': {
        'train': '/cluster/work/cvl/haoqin/datasets/matting-data/Backgrounds/train',
        'valid': '/cluster/work/cvl/haoqin/datasets/matting-data/Backgrounds/valid',
    },
    'background_videos': {
        'train': '/cluster/work/cvl/haoqin/datasets/matting-data/BackgroundVideos/train',
        'valid': '/cluster/work/cvl/haoqin/datasets/matting-data/BackgroundVideos/valid',
    },
    
    
    'coco_panoptic': {
        'imgdir': '/cluster/work/cvl/haoqin/datasets/matting-data/coco/train2017',
        'anndir': '/cluster/work/cvl/haoqin/datasets/matting-data/coco/annotations/panoptic_train2017',
        'annfile': '/cluster/work/cvl/haoqin/datasets/matting-data/coco/annotations/panoptic_train2017.json',
    },
    'spd': {
        'imgdir': '/cluster/work/cvl/haoqin/datasets/matting-data/SuperviselyPersonDataset/img',
        'segdir': '/cluster/work/cvl/haoqin/datasets/matting-data/SuperviselyPersonDataset/seg',
    },
    'youtubevis': {
        'videodir': '/cluster/work/cvl/haoqin/datasets/matting-data/YouTubeVIS/train/JPEGImages',
        'annfile': '/cluster/work/cvl/haoqin/datasets/matting-data/YouTubeVIS/train/instances.json',
    }
    
}
