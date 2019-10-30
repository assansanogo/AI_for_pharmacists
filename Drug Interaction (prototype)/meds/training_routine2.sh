#!/usr/bin/env bash


cd /raid/data2/asanogo/meds


/cyclope/allanza/darknet_pjreddie3/darknet detector train /raid/data2/asanogo/meds/obj_meds2.data  \
/raid/data2/asanogo/meds/cfg/yolov2.cfg \
/raid/data2/asanogo/meds/backup2/darknet19_448.conv.23
-dont_show -mjpeg_port 8020 -map \
-gpus=3
