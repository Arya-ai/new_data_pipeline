import subprocess,imageio
import os
from tqdm import tqdm
path = "/home/intern/video_caption/video_captions/data/MSR-VTT/TrainValVideo/"
frames = []
files = os.listdir(path)
out = open("/home/intern/video_caption/video_captions/data/MSR-VTT/frame_count.txt","w")
count = 0
for i in tqdm(files):
	#cmd ="ffprobe -select_streams v -show_streams "+i+" 2>/dev/null | grep nb_frames | sed -e 's/nb_frames=//'"
 # 	cmd ="ffprobe -select_streams v -show_streams "+i+" 2>/dev/null | grep nb_frames | sed -e 's/nb_frames=//'"	
	# print cmd
	# ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
	# print ps.communicate()
	#frames.append(int(ps.communicate()[0].strip('\n')))
	vid = imageio.get_reader(path+i,"ffmpeg")
	out.write(str(vid._meta["nframes"])+"\n")
	# print "Done with {0}".format(i)
out.close()

