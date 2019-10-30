#!/usr/bin/env bash


cd /raid/data2/asanogo/meds


/cyclope/allanza/darknet_pjreddie3/darknet detector train /raid/data2/asanogo/meds/obj_meds.data  \
/raid/data2/asanogo/meds/cfg/yolov3.cfg \
/raid/data2/asanogo/meds/backup/darknet53.conv.74
-dont_show -mjpeg_port 8010 -map \
-gpus=2
