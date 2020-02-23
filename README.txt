* The code structure is show as below:

.
├── README
├── __init__.py
├── datasets
│   ├── __init__.py
│   └── svhn.py
├── graded_images
│   ├── 1.png
│   ├── 10.png
│   ├── 2.png
│   ├── 3.png
│   ├── 4.png
│   ├── 5.png
│   ├── 6.png
│   ├── 7.png
│   ├── 8.png
│   ├── 9.png
│   └── video1.mp4
├── inference.py
├── main.py
├── models
│   ├── __init__.py
│   ├── my_net.py
│   └── my_vgg16.py
├── run.py
├── run_video.py
├── svhn
│   └── 04-18-00-50
│       └── Netmy_net_solveradam_epochs800_batchSize1024_lr0.0001_batchnormTrue
│           ├── checkpoint.pth.tar
│           ├── model_best.pth.tar
│           ├── netStruct
│           │   └── events.out.tfevents.1555573820.xi-home-desktop
│           ├── test
│           │   └── events.out.tfevents.1555573820.xi-home-desktop
│           └── train
│               └── events.out.tfevents.1555573820.xi-home-desktop
├── test_input
│   ├── img1.jpg
│   ├── img10.jpg
│   ├── img2.jpg
│   ├── img3.jpg
│   ├── img4.jpg
│   ├── img5.jpg
│   ├── img6.jpg
│   ├── img7.jpg
│   ├── img8.jpg
│   ├── img9.jpg
│   └── video1.mp4
├── train.py
└── utils
    ├── __init__.py
    ├── average_meter.py
    ├── copy_files.py
    ├── crop_bkg.py
    ├── crop_digit.py
    ├── crop_digit_h5py.py
    ├── download_flickr.py
    ├── listdataset.py
    ├── opencv_mser.py
    ├── transform.py
    └── util.py

* To run the run.py. do:
>> python run.py

* 
Presentation 
YouTube Link:
1. https://youtu.be/T8JR1SBmhGs 
2. https://youtu.be/MUVcTpa0LkY

Demo Youtube Link:
1. https://youtu.be/_iE2LqsT09s
2. https://youtu.be/4AlwVn8JX5Q
