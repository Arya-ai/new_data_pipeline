import lmdb
import numpy as np
import cv2	#openCV for getting image attributes
import os, sys
from datum_pb2 import Datum
import pandas as pd
from threading import Thread
from Queue import Queue

class readWorker():
	'''
	A thread for reading the data
	'''
	# imageNumber: number of image read (in case of multiple subdirs of labeldirs)
	def readImage(self, fileQueue, data_dir, nInputPerRecord):
		'''
		Reads the files and pushes them into the file queue
		'''
		# while True:
		if nInputPerRecord == 1:
			'''
			Directly read the images from each subdirectory (label directory)
			'''
			imageNumber = 0
			labeldirs = [os.path.join(data_dir, subdir) for subdir in os.listdir(data_dir)]
			for labeldir in enumerate(labeldirs):
				for key, image in enumerate(sorted([os.listdir(labeldir)])):
					imagePath = os.path.join(labeldir, image)
					ndarray = cv2.imread(imagePath)
					slabel = imagePath.split('/')[-2]
					item = tuple([ndarray, slabel, imageNumber, key])
					fileQueue.put(item)		# pass imagepath along with class label
		else:
			'''
			Read the images from each subdirectory (label directory) of each subdirectory
			'''
			subdirs = [os.path.abspath(os.path.join(data_dir, subdir)) for subdir in os.listdir(data_dir)]

			# test
			# try:
			# 	print "\nInside readImage, subdirs:", len(subdirs)
			# 	print "\nnInputPerRecord:", nInputPerRecord
			# 	assert nInputPerRecord == len(subdirs)
			# except AssertionError:
			# 	print "Error: Number of Images per record does not match number of subdirectories"
			# 	return

			ssdirs = [os.listdir(subdir) for subdir in subdirs]

			labeldirs = []

			for ssdir, subdir in zip(ssdirs, subdirs):
				labeldirs.append([os.path.join(subdir, ssd) for ssd in ssdir])

			images = []

			key = 0

			for label_list in [list(i) for i in zip(*labeldirs)]:
				images = zip(*[sorted(os.listdir(labeldir)) for labeldir in label_list])
				print images

				imageNumber = nInputPerRecord
				for image in images:
					key += 1
					for im, rootPath in zip(image, label_list):
						imagePath = os.path.join(rootPath, im)
						try:
							ndarray = cv2.imread(imagePath)
						except IOError as e:
							print "Error:", e
						slabel = imagePath.split('/')[-2]
						item = tuple([ndarray, slabel, imageNumber, key])		# pass imagepath along with class label
						print "Pushing item into File Queue..."
						fileQueue.put(item)
						imageNumber -= 1		# decrement for the thread to distinguish

	def readNumeric(self, fileQueue, file, labels):
		while True:
			if file.endswith(".csv"):
				try:
					df = pd.read_csv(file)
				except IOError as e:
					print "Error:", e
			elif file.endswith(".json"):
				try:
					df = pd.read_json(file)
				except IOError as e:
					print "Error:", e
			else:
				print "Error: Provide the file in valid format (.csv or .json)"
				return

			labelSeries = []
			for label in labels:
				labelSeries.append(df.pop(label))

			labeldf = pd.concat(labelSeries, axis=1)		# concat all the labels into a single df

			for idx in xrange(len(df)):
				datum = df.iloc[idx]
				datum = datum.to_frame()
				datum = datum.to_records(index=False)

				label = labeldf.iloc[idx]
				label = label.to_frame()
				label = label.to_records(index=False)

				item = tuple([datum, label, idx])
				print "Pushing item into File Queue..."
				fileQueue.put(item)

	def readText(self, fileQueue, file):
		while True:
			return None

class datumWorker():
	'''
	A thread for preparing the datum
	'''
	def ImageDatum(self, fileQueue, datumQueue):
		count = 0
		while True:
			ndarray, slabel, imageNumber, key = list(fileQueue.get())
			count += 1
			dims = list(ndarray.shape)

			datum = Datum()

			labelDatum = datum.classs
			labelDatum.identifier = str(key)
			labelDatum.slabel = slabel

			imageDatum = datum.imgdata.add()
			imageDatum.identifier = str(key)
			imageDatum.channels = dims[2]
			imageDatum.height = dims[0]
			imageDatum.width = dims[1]

			# testing
			print "datum #{}".format(count)
			print "\ndatum:\n", datum

			imageDatum.data = ndarray.tobytes()

			item = tuple([imageDatum, labelDatum, imageNumber, key])
			datumQueue.put(item)
			fileQueue.task_done()		# let the fileQueue know item has been processed and is safe to delete

	def NumericDatum(self, fileQueue, datumQueue):
		count = 0
		while True:
			datum, label, key = list(fileQueue.get())
			count += 1
			datum = Datum()
			labelDatum = datum.classs
			labelDatum.identifier = str(key)
			labelDatum.nlabel = label

			numericDatum = datum.numeric
			numericDatum.identifier = str(key)
			numericDatum.size.dim = label.shape[0]

			# testing
			print "datum #{}".format(count)
			print "\ndatum:\n", datum

			numericDatum.data = datum.tobytes()

			item = tuple([numericDatum, labelDatum, key])
			datumQueue.put(item)
			fileQueue.task_done()

	def TextDatum(item):
		while  True:
			return None


class writeWorker():
	def __init__(self, datumQueue, env, dbQueue):
		while True:
			item = list(datumQueue.get())
			dbId = 0		# which db to push to in case of multiple inputs per record
			if len(item) == 4:
				datum, label, dbId, key = item
			else:
				datum, label, key = item

			if dbId == 0:
				with env.begin(write=True) as txn:
					txn.put(str(key).encode('ascii'), datum.SerializeToString())

				labeldbHandle = dbList[-1]
				with env.begin(write=True, db=labeldbHandle) as txn:
					txn.put(str(key).encode('ascii'), label.SerializeToString())
			else:
				dbHandle = dbList[dbId - 1]
				with env.begin(write=True, db=dbHandle) as txn:
					txn.put(str(key).encode('ascii'), datum.SerializeToString())




if __name__ == '__main__':
	args = sys.argv[1:]		# clip off the script name
	if len(args) <= 1:
		print "Error: source directory not specified"
		print "Usage: python serialize.py [--image || --numeric || --text] [nPerRecord] [data_dir || file ..]"
		sys.exit(-1)
	else:
		dataType = args[0]
		nInputPerRecord = int(args[1])
		data_dir = args[2]

		# config: 2 queues, 3 workers
		fileQueue = Queue()
		datumQueue = Queue()
		readWorker = readWorker()
		datumWorker = datumWorker()
		# writeWorker = writeWorker()

		env = lmdb.open('lmdb/datumdb0', max_dbs=10)
		dbList = []
		# create dbs for datums
		for i in xrange(1,nInputPerRecord):
			dbName = 'datumdb' + str(i)
			dbList.append(env.open_db(dbName))
		# create labeldb
		dbList.append(env.open_db('labeldb'))

		if dataType == '--image':
			read_worker = Thread(target=readWorker.readImage, args=(fileQueue, data_dir, nInputPerRecord))
			datum_worker = Thread(target=datumWorker.ImageDatum, args=(fileQueue, datumQueue,))
		elif dataType == '--numeric':
			labels = [str(i) for i in raw_input("Mention the output labels: ")]
			read_worker = Thread(target=readWorker.readNumeric(fileQueue, data_dir, labels))
			datum_worker = Thread(target=datumWorker.NumericDatum, args=(fileQueue, datumQueue,))
		elif dataType == '--text':
			read_worker = Thread(target=readWorker.readText, args=(fileQueue,))
			datum_worker = Thread(target=datumWorker.TextDatum, args=(fileQueue, datumQueue,))
		else:
			print "Error: Incorrect or no tag given"
			print "Usage: python serialize.py [--image || --numeric || --text] [data_dir || file ..]"
			sys.exit(-1)

		write_worker = Thread(target=writeWorker, args=(datumQueue, env, dbList))

		read_worker.setDaemon(True)
		read_worker.start()
		print "Read Worker started"
		datum_worker.setDaemon(True)
		datum_worker.start()
		print "Datum Worker started"
		write_worker.setDaemon(True)
		write_worker.start()
		print "Write Worker started"

		# fileQueue.join()
		# datumQueue.join()

		# env.close()

		print "Hajime (https://translate.google.com/#ja/en/Hajime)"











def serializeImage(data_dir, env):
	with env.begin(write=True) as txn:
		key = 0
	
		n = int(raw_input("How many input images per training record? : "))
		if True:
			return
		else:
			'''
			Read the images from each subdirectory (label directory) of each subdirectory
			'''
			subdirs = [os.path.abspath(os.path.join(data_dir, subdir)) for subdir in os.listdir(data_dir)]

			ssdirs = [os.listdir(subdir) for subdir in subdirs]

			labeldirs = []

			for ssdir, subdir in zip(ssdirs, subdirs):
				labeldirs.append([os.path.join(subdir, ssd) for ssd in ssdir])

			# Class of the image
			# took the first class to be zero (can be changed to 1)
			Class = 0
			images = []

			for label_list in [list(i) for i in zip(*labeldirs)]:
				images = zip(*[os.listdir(labeldir) for labeldir in label_list])

				image_paths = []

				for idx, image in enumerate(images):
					image_paths = []
					for im, rootPath in zip(image, label_list):
						image_paths.append(os.path.join(rootPath, im))
					
					datum = Datum()
					LabelDatum = datum.classs
					LabelDatum.identifier = str(idx)
					LabelDatum.slabel = image_paths[0].split('/')[-2]

					for image_path in image_paths:
						im = cv2.imread(image_path)
						dims = list(im.shape)
						ImageDatum = datum.imgdata.add()
						ImageDatum.identifier = str(idx)
						ImageDatum.channels = dims[2]
						ImageDatum.height = dims[0]
						ImageDatum.width = dims[1]
						ImageDatum.data = im.tobytes()

					str_id = '{:08}'.format(idx)
					txn.put(str_id.encode('ascii'), datum.SerializeToString())

	# close the environment
	env.close()


def SerializeNumeric(file):

	# works for single labels
	# have to work on multiple labels

	env = lmdb.open('lmdb/datumdb', max_dbs=2)
	labeldb = env.open_db('labeldb')

	# ask for the columns which are labels
	labels = [str(x) for x in raw_input("Labels (sep by commas): ").split()]

	# load the data
	df = pd.read_csv(file)
	labeldf = df.pop(labels[0])		# separate the label

	for idx in xrange(len(df)):
		print "Iter #{}".format(idx)
		datum = Datum()
		LabelDatum = datum.classs
		LabelDatum.identifier = str(idx)
		LabelDatum.nlabel = labeldf[idx]

		NumericDatum = datum.numeric
		NumericDatum.identifier = str(idx)

		x = df.iloc[idx]
		x = x.to_frame()
		x = x.to_records(index=False)

		NumericDatum.size.dim = len(labels)
		NumericDatum.data = x.tobytes()

		str_id = '{:08}'.format(idx)

		# push the label to labeldb
		with env.begin(write=True, db=labeldb) as txn:
			print "begin label #{}".format(idx)
			txn.put(str_id.encode('ascii'), LabelDatum.SerializeToString())
			print "end label #{}".format(idx)

		# push the data to datadb
		with env.begin(write=True) as txn:
			print "begin datum #{}".format(idx)
			txn.put(str_id.encode('ascii'), NumericDatum.SerializeToString())
			print "end datum #{}".format(idx)

	# close the environment
	env.close()