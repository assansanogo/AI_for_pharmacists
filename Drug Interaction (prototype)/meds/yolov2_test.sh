#!/usr/bin/env bash

cd /raid/data2/asanogo/meds


/cyclope/allanza/darknet_pjreddie3/darknet detector test /raid/data2/asanogo/meds/obj_meds2.data  \
/raid/data2/asanogo/meds/cfg/yolov2_test.cfg \
/raid/data2/asanogo/meds/backup2/yolov2_110000.weights \
/raid/data2/asanogo/meds/object_detector/paracetamol_thesis1/paracetamol3/vid_frame10.jpg -i 0 -thresh 0.9

