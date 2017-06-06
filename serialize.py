import lmdb
import numpy as np
import cv2  #openCV for getting image attributes
import os, sys
from datum_pb2 import Datum
import pandas as pd
from threading import Thread
from multiprocessing import Manager
from Queue import Queue
from sklearn.feature_extraction.text import CountVectorizer
import json
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

manager = Manager()

class readWorker():
    '''
    A worker for reading the data
    '''
    def readImage(self, fileQueue, data_dir, nInputPerRecord, readFlag):
        logger.debug("nInputPerRecord: " + str(nInputPerRecord))
        if nInputPerRecord == 1:
            '''
            Directly read the images from each subdirectory (label directory)
            '''
            imageNumber = 0
            labeldirs = [os.path.join(data_dir, subdir) for subdir in os.listdir(data_dir)]

            key = 0

            for labeldir in labeldirs:
                for image in os.listdir(labeldir):
                    key += 1
                    imagePath = os.path.join(labeldir, image)
                    ndarray = cv2.imread(imagePath)
                    slabel = imagePath.split('/')[-2]
                    item = tuple([ndarray, slabel, imageNumber, key])
                    logger.debug("Pushed item {} into File Queue...".format(key))
                    fileQueue.put(item)     # pass imagepath along with class label

            readFlag.value = 1
        else:
            '''
            Read the images from each subdirectory (label directory) of each subdirectory
            '''
            subdirs = [os.path.abspath(os.path.join(data_dir, subdir)) for subdir in os.listdir(data_dir)]

            # test
            try:
                assert nInputPerRecord == len(subdirs)
            except AssertionError:
                logger.error("Number of Images per record does not match number of subdirectories")
                sys.exit(-1)

            ssdirs = [os.listdir(subdir) for subdir in subdirs]

            labeldirs = []

            for ssdir, subdir in zip(ssdirs, subdirs):
                labeldirs.append([os.path.join(subdir, ssd) for ssd in ssdir])

            images = []

            key = 0

            for label_list in [list(i) for i in zip(*labeldirs)]:
                images = zip(*[sorted(os.listdir(labeldir)) for labeldir in label_list])

                imageNumber = nInputPerRecord
                for image in images:
                    key += 1
                    for im, rootPath in zip(image, label_list):
                        imagePath = os.path.join(rootPath, im)

                        try:
                            ndarray = cv2.imread(imagePath)
                        except IOError as e:
                            logger.error("Error:", e)
                        slabel = imagePath.split('/')[-2]
                        item = tuple([ndarray, slabel, imageNumber, key])       # pass imagepath along with class label
                        logger.debug("ReadWorker: Pushed item {} into File Queue...".format(key))
                        fileQueue.put(item)
                        imageNumber -= 1        # decrement for the thread to distinguish

            readFlag.value = 1

    def readNumeric(self, fileQueue, data_dir, options, readFlag):
        label = options['label']
        file = os.path.join(data_dir, os.listdir(data_dir)[0])
        if file.endswith('.csv'):
            try:
                df = pd.read_csv(file)
            except IOError as e:
                logger.error("Error reading csv file: ", exc_info=True)
        elif file.endswith('.json'):
            try:
                df = json.load(open(file))
            except IOError as e:
                logger.error("Error opening json file: ", exc_info=True)
        else:
            logger.error("Error: Provide the file in valid format (.csv or .json)")
            sys.exit(-1)

        # labelSeries = []
        # for label in labels:
        #     labelSeries.append(df.pop(label))

        # labeldf = pd.concat(labelSeries, axis=1)        # concat all the labels into a single df
        labeldf = df.pop(label)

        for idx in xrange(len(df)):
            data = df.iloc[idx].to_frame.to_records(index=False)
            '''
            this will return a numpy record array which needs to be converted to a numpy array
            for it to be converted back from a byte array
            '''
            # retrieve the datatype
            dt = data.dtype.fields[u'0'][0]
            # convert into numpy array
            np_data = np.array(data.view(dt))

            label = labeldf.iloc[idx]       # for single label
            # label = label.to_frame()
            # label = label.to_records(index=False)

            item = tuple([np_data, label, idx])
            logger.debug("Pushed item {} into File Queue...".format(idx + 1))
            fileQueue.put(item)

        readFlag.value = 1

    def readText(self, fileQueue, data_dir, options, readFlag):
        name = options['name']
        text = options['text']
        label = options['label']
        file = os.listdir(data_dir)[0]
        if file.endswith(".csv"):
            try:
                df = pd.read_csv(file)
            except IOError as e:
                logger.error("Error:", e)
        elif file.endswith(".json"):
            try:
                df = json.load(open(file))
            except IOError as e:
                logger.error("Error:", e)
        else:
            logger.error("Error: Provide the file in valid format (.csv or .json)")
            return

        vectorizer = CountVectorizer(token_pattern=u"(?u)\\b\\w+\\b")

        caps = []
        for i in df[name]:
            caps.append(i[text])

        train_features = vectorizer.fit_transform(caps)
        vecs = train_features.toarray()

        for idx in xrange(len(df[name])):
            textDatum = vecs[idx]
            textLabel = df[name][idx][label]
            item = tuple([textDatum, textLabel, idx])
            logger.debug("Pushed item {} into File Queue...".format(idx + 1))
            fileQueue.put(item)

        readFlag.value = 1

class datumWorker():
    '''
    A worker for preparing the datum
    '''
    def ImageDatum(self, fileQueue, datumQueue):
        count = 0
        while True:
            data, slabel, imageNumber, key = list(fileQueue.get())
            count += 1
            dims = list(data.shape)

            datum = Datum()

            labelDatum = datum.classs
            labelDatum.identifier = str(key)
            labelDatum.slabel = slabel

            imageDatum = datum.imgdata.add()
            imageDatum.identifier = str(key)
            imageDatum.channels = dims[2]
            imageDatum.height = dims[0]
            imageDatum.width = dims[1]
            imageDatum.data = data.tobytes()

            item = tuple([imageDatum, labelDatum, imageNumber, key])
            datumQueue.put(item)
            logger.debug("DatumWorker: Pushed datum #{} into Datum Queue".format(count))
            fileQueue.task_done()       # let the fileQueue know item has been processed and is safe to delete

    def NumericDatum(self, fileQueue, datumQueue):
        count = 0
        while True:
            data, label, key = list(fileQueue.get())
            count += 1
            datum = Datum()
            labelDatum = datum.classs
            labelDatum.identifier = str(key)
            labelDatum.nlabel = label

            numericDatum = datum.numeric
            numericDatum.identifier = str(key)
            # numericDatum.size.dim = label.shape[0]
            numericDatum.size.dim = 1       # currently just for 1 label
            numericDatum.data = data.tobytes()

            item = tuple([numericDatum, labelDatum, key])
            datumQueue.put(item)
            logger.debug("Pushed datum #{} into Datum Queue".format(count))
            fileQueue.task_done()

    def TextDatum(self, fileQueue, datumQueue):
        count = 0
        while  True:
            data, label, key = list(fileQueue.get())
            count += 1

            datum = Datum()
            textDatum = datum.numeric
            textDatum.identifier = str(key)
            textDatum.size.dim = 1
            textDatum.data = np.array(data).tobytes()

            labelDatum = datum.classs
            labelDatum.identifier = str(key)
            labelDatum.nlabel = label

            item = tuple([textDatum, labelDatum, key])
            datumQueue.put(item)
            logger.debug("Pushed datum #{} into Datum Queue".format(count))
            fileQueue.task_done()
            
class writeWorker():
    '''
    A worker for writing the datum to lmdb database
    '''
    def __init__(self, datumQueue, env, dbList):
        while True:
            item = list(datumQueue.get())
            dbId = 0        # which db to push to in case of multiple inputs per record
            if len(item) == 4:
                datum, label, dbId, key = item
            else:
                datum, label, key = item

            if dbId == 0:
                with env.begin(write=True) as txn:
                    txn.put(str(key).encode('ascii'), datum.SerializeToString())

                labelDBHandle = dbList[-1]
                with env.begin(write=True, db=labelDBHandle) as txn:
                    txn.put(str(key).encode('ascii'), label.SerializeToString())
            else:
                dbHandle = dbList[dbId - 1]
                with env.begin(write=True, db=dbHandle) as txn:
                    txn.put(str(key).encode('ascii'), datum.SerializeToString())

            logger.debug("WriteWorker: Datum #{} written to lmdb".format(key))


class Serialize():

    def __init__(self, nInputPerRecord):
        self.nInputPerRecord = int(nInputPerRecord)
        # config: 2 queues, 3 workers
        self.fileQueue = Queue()
        self.datumQueue = Queue()
        # dbs for each input per record + 1 for label
        self.env = lmdb.open('lmdb/datumdb0', max_dbs=nInputPerRecord + 1)
        self.readFlag = manager.Value('i', 0)
        self.doneFlag = manager.Value('i', 0)
        logger.debug("Serialize Instance created with {} dbs".format(nInputPerRecord + 1))

    def writeToLmdb(self, options):
        data_dir, args = options
        logger.debug("Data directory: " + str(data_dir))
        logger.info("Writing to LMDB")
        dataType = args['dataType'][0]

        dbList = []
        # create dbs for datums
        for i in xrange(1,self.nInputPerRecord):
            dbName = 'datumdb' + str(i)
            dbList.append(self.env.open_db(dbName))
        # create labeldb
        dbList.append(self.env.open_db('labeldb'))

        if dataType == 'image':
            self.read_worker = Thread(target=readWorker().readImage, args=(self.fileQueue, data_dir, self.nInputPerRecord, self.readFlag))
            self.datum_worker = Thread(target=datumWorker().ImageDatum, args=(self.fileQueue, self.datumQueue,))

        elif dataType == 'numeric':
            # labels = [str(label) for label in args['labels'][0].split(' ')]
            label = args['label'][0]
            options = {'label': label}
            self.read_worker = Thread(target=readWorker().readNumeric(self.fileQueue, data_dir, options, self.readFlag))
            self.datum_worker = Thread(target=datumWorker().NumericDatum, args=(self.fileQueue, self.datumQueue,))

        elif dataType == 'text':
            name = str(args['name']).strip()
            text = str(args['text']).strip()
            label = str(args['label']).strip()
            options = {'name': name, 'text': text, 'label': label}
            self.read_worker = Thread(target=readWorker().readText, args=(self.fileQueue, data_dir, options, self.readFlag))
            self.datum_worker = Thread(target=datumWorker().TextDatum, args=(self.fileQueue, self.datumQueue,))

        else:
            logger.error(" :Incorrect or no tag given. Specify dataType in request.")
            sys.exit(-1)

        self.write_worker = Thread(target=writeWorker, args=(self.datumQueue, self.env, dbList))

        self.read_worker.setDaemon(True)
        self.read_worker.start()
        logger.debug("Read Worker started")
        self.datum_worker.setDaemon(True)
        self.datum_worker.start()
        logger.debug("Datum Worker started")
        self.write_worker.setDaemon(True)
        self.write_worker.start()
        logger.debug("Write Worker started")

        print "Hajime (https://translate.google.com/#ja/en/Hajime)"


if __name__ == '__main__':
    args = sys.argv[1:]     # clip off the script name
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
            print "A typical json file body:\n" + \
            "{\n" + \
            "\t'name': [\n" + \
            "\t\t{\n" + \
            "\t\t\t'text': 'sample input text'\n" + \
            "\t\t\t'label': 'label for above text'\n" + \
            "},\n" + \
            "\t\t{\n" + \
            "\t\t\t'text': ' ... '\n" + \
            "\t\t\t'label': '' ... '\n" + \
            "},\n" + \
            "\t\t... \n" + \
            "\t]\n" + \
            "}"
            
            print "Enter the fields accordingly."

            name = str(raw_input("Enter the name: ").strip())
            text = str(raw_input("Enter the input: ").strip())
            label = str(raw_input("Enter the label: ").strip())
            read_worker = Thread(target=readWorker.readText, args=(fileQueue,data_dir,name,text,label))
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

        print "Hajime (https://translate.google.com/#ja/en/Hajime)"
