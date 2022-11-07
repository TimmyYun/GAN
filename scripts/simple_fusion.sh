wav1=27
wav2=28
wav3=29
rgb1=21
rgb2=22
rgb3=23
thr1=135
thr2=136
thr3=137
wavrgb1=1
wavrgb2=2
wavrgb3=3
wavrgbthr1=71
wavrgbthr2=72
wavrgbthr3=73
echo "WAV+RGB"
python /raid/timur_unaspekov/projects/trimodal_person_verification/simple_fusion.py --path_wav /raid/timur_unaspekov/meta/results/wav/exp$wav1/result/test_predictons_veer_vc_400.txt --path_rgb /raid/timur_unaspekov/meta/results/rgb/exp$rgb1/result/test_predictons_veer_vc_400.txt

python /raid/timur_unaspekov/projects/trimodal_person_verification/simple_fusion.py --path_wav /raid/timur_unaspekov/meta/results/wav/exp$wav2/result/test_predictons_veer_vc_400.txt --path_rgb /raid/timur_unaspekov/meta/results/rgb/exp$rgb2/result/test_predictons_veer_vc_400.txt

python /raid/timur_unaspekov/projects/trimodal_person_verification/simple_fusion.py --path_wav /raid/timur_unaspekov/meta/results/wav/exp$wav3/result/test_predictons_veer_vc_400.txt --path_rgb /raid/timur_unaspekov/meta/results/rgb/exp$rgb3/result/test_predictons_veer_vc_400.txt
echo "WAV+RGB+THR"
python /raid/timur_unaspekov/projects/trimodal_person_verification/simple_fusion.py --path_wav /raid/timur_unaspekov/meta/results/wav/exp$wav1/result/test_predictons_veer_vc_400.txt --path_rgb /raid/timur_unaspekov/meta/results/rgb/exp$rgb1/result/test_predictons_veer_vc_400.txt  --path_thr  /raid/timur_unaspekov/meta/results/thr/exp$thr1/result/test_predictons_veer_vc_400.txt

python /raid/timur_unaspekov/projects/trimodal_person_verification/simple_fusion.py --path_wav /raid/timur_unaspekov/meta/results/wav/exp$wav1/result/test_predictons_veer_vc_400.txt --path_rgb /raid/timur_unaspekov/meta/results/rgb/exp$rgb2/result/test_predictons_veer_vc_400.txt  --path_thr  /raid/timur_unaspekov/meta/results/thr/exp$thr2/result/test_predictons_veer_vc_400.txt

python /raid/timur_unaspekov/projects/trimodal_person_verification/simple_fusion.py --path_wav /raid/timur_unaspekov/meta/results/wav/exp$wav3/result/test_predictons_veer_vc_400.txt --path_rgb /raid/timur_unaspekov/meta/results/rgb/exp$rgb3/result/test_predictons_veer_vc_400.txt  --path_thr  /raid/timur_unaspekov/meta/results/thr/exp$thr3/result/test_predictons_veer_vc_400.txt
echo "WAV+RGB+WAVRGB"
python /raid/timur_unaspekov/projects/trimodal_person_verification/simple_fusion.py --path_wav /raid/timur_unaspekov/meta/results/wav/exp$wav1/result/test_predictons_veer_vc_400.txt --path_rgb /raid/timur_unaspekov/meta/results/rgb/exp$rgb1/result/test_predictons_veer_vc_400.txt --path_bi /raid/timur_unaspekov/meta/results/wavrgb/exp$wavrgb1/result/test_predictons_veer_vc_400.txt

python /raid/timur_unaspekov/projects/trimodal_person_verification/simple_fusion.py --path_wav /raid/timur_unaspekov/meta/results/wav/exp$wav2/result/test_predictons_veer_vc_400.txt --path_rgb /raid/timur_unaspekov/meta/results/rgb/exp$rgb2/result/test_predictons_veer_vc_400.txt --path_bi /raid/timur_unaspekov/meta/results/wavrgb/exp$wavrgb2/result/test_predictons_veer_vc_400.txt

python /raid/timur_unaspekov/projects/trimodal_person_verification/simple_fusion.py --path_wav /raid/timur_unaspekov/meta/results/wav/exp$wav3/result/test_predictons_veer_vc_400.txt --path_rgb /raid/timur_unaspekov/meta/results/rgb/exp$rgb3/result/test_predictons_veer_vc_400.txt --path_bi /raid/timur_unaspekov/meta/results/wavrgb/exp$wavrgb3/result/test_predictons_veer_vc_400.txt
echo "WAV+RGB+THR+WAVRGB"
python /raid/timur_unaspekov/projects/trimodal_person_verification/simple_fusion.py --path_wav /raid/timur_unaspekov/meta/results/wav/exp$wav1/result/test_predictons_veer_vc_400.txt --path_rgb /raid/timur_unaspekov/meta/results/rgb/exp$rgb1/result/test_predictons_veer_vc_400.txt --path_thr  /raid/timur_unaspekov/meta/results/thr/exp$thr1/result/test_predictons_veer_vc_400.txt --path_bi /raid/timur_unaspekov/meta/results/wavrgb/exp$wavrgb1/result/test_predictons_veer_vc_400.txt

python /raid/timur_unaspekov/projects/trimodal_person_verification/simple_fusion.py --path_wav /raid/timur_unaspekov/meta/results/wav/exp$wav2/result/test_predictons_veer_vc_400.txt --path_rgb /raid/timur_unaspekov/meta/results/rgb/exp$rgb2/result/test_predictons_veer_vc_400.txt --path_thr  /raid/timur_unaspekov/meta/results/thr/exp$thr2/result/test_predictons_veer_vc_400.txt --path_bi /raid/timur_unaspekov/meta/results/wavrgb/exp$wavrgb2/result/test_predictons_veer_vc_400.txt

python /raid/timur_unaspekov/projects/trimodal_person_verification/simple_fusion.py --path_wav /raid/timur_unaspekov/meta/results/wav/exp$wav3/result/test_predictons_veer_vc_400.txt --path_rgb /raid/timur_unaspekov/meta/results/rgb/exp$rgb3/result/test_predictons_veer_vc_400.txt --path_thr  /raid/timur_unaspekov/meta/results/thr/exp$thr3/result/test_predictons_veer_vc_400.txt --path_bi /raid/timur_unaspekov/meta/results/wavrgb/exp$wavrgb3/result/test_predictons_veer_vc_400.txt
echo "WAV+RGB+WAVRGB+WAVRGBTHR"
python /raid/timur_unaspekov/projects/trimodal_person_verification/simple_fusion.py --path_wav /raid/timur_unaspekov/meta/results/wav/exp$wav1/result/test_predictons_veer_vc_400.txt --path_rgb /raid/timur_unaspekov/meta/results/rgb/exp$rgb1/result/test_predictons_veer_vc_400.txt --path_bi /raid/timur_unaspekov/meta/results/wavrgb/exp$wavrgb1/result/test_predictons_veer_vc_400.txt --path_tri /raid/timur_unaspekov/meta/results/wavrgbthr/exp$wavrgbthr1/result/test_predictons_veer_vc_400.txt

python /raid/timur_unaspekov/projects/trimodal_person_verification/simple_fusion.py --path_wav /raid/timur_unaspekov/meta/results/wav/exp$wav2/result/test_predictons_veer_vc_400.txt --path_rgb /raid/timur_unaspekov/meta/results/rgb/exp$rgb2/result/test_predictons_veer_vc_400.txt --path_bi /raid/timur_unaspekov/meta/results/wavrgb/exp$wavrgb2/result/test_predictons_veer_vc_400.txt --path_tri /raid/timur_unaspekov/meta/results/wavrgbthr/exp$wavrgbthr2/result/test_predictons_veer_vc_400.txt

python /raid/timur_unaspekov/projects/trimodal_person_verification/simple_fusion.py --path_wav /raid/timur_unaspekov/meta/results/wav/exp$wav3/result/test_predictons_veer_vc_400.txt --path_rgb /raid/timur_unaspekov/meta/results/rgb/exp$rgb3/result/test_predictons_veer_vc_400.txt --path_bi /raid/timur_unaspekov/meta/results/wavrgb/exp$wavrgb3/result/test_predictons_veer_vc_400.txt --path_tri /raid/timur_unaspekov/meta/results/wavrgbthr/exp$wavrgbthr3/result/test_predictons_veer_vc_400.txt
echo "WAV+RGB+THR+WAVRGB+WAVRGBTHR"
python /raid/timur_unaspekov/projects/trimodal_person_verification/simple_fusion.py --path_wav /raid/timur_unaspekov/meta/results/wav/exp$wav1/result/test_predictons_veer_vc_400.txt --path_rgb /raid/timur_unaspekov/meta/results/rgb/exp$rgb1/result/test_predictons_veer_vc_400.txt  --path_thr  /raid/timur_unaspekov/meta/results/thr/exp$thr1/result/test_predictons_veer_vc_400.txt --path_bi /raid/timur_unaspekov/meta/results/wavrgb/exp$wavrgb1/result/test_predictons_veer_vc_400.txt --path_tri /raid/timur_unaspekov/meta/results/wavrgbthr/exp$wavrgbthr1/result/test_predictons_veer_vc_400.txt

python /raid/timur_unaspekov/projects/trimodal_person_verification/simple_fusion.py --path_wav /raid/timur_unaspekov/meta/results/wav/exp$wav2/result/test_predictons_veer_vc_400.txt --path_rgb /raid/timur_unaspekov/meta/results/rgb/exp$rgb2/result/test_predictons_veer_vc_400.txt  --path_thr  /raid/timur_unaspekov/meta/results/thr/exp$thr2/result/test_predictons_veer_vc_400.txt --path_bi /raid/timur_unaspekov/meta/results/wavrgb/exp$wavrgb2/result/test_predictons_veer_vc_400.txt --path_tri /raid/timur_unaspekov/meta/results/wavrgbthr/exp$wavrgbthr2/result/test_predictons_veer_vc_400.txt

python /raid/timur_unaspekov/projects/trimodal_person_verification/simple_fusion.py --path_wav /raid/timur_unaspekov/meta/results/wav/exp$wav3/result/test_predictons_veer_vc_400.txt --path_rgb /raid/timur_unaspekov/meta/results/rgb/exp$rgb3/result/test_predictons_veer_vc_400.txt  --path_thr  /raid/timur_unaspekov/meta/results/thr/exp$thr3/result/test_predictons_veer_vc_400.txt --path_bi /raid/timur_unaspekov/meta/results/wavrgb/exp$wavrgb3/result/test_predictons_veer_vc_400.txt --path_tri /raid/timur_unaspekov/meta/results/wavrgbthr/exp$wavrgbthr3/result/test_predictons_veer_vc_400.txt