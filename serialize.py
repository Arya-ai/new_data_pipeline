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
                    item = tuple(['image', ndarray, slabel, imageNumber, key])
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
                        item = tuple(['image', ndarray, slabel, imageNumber, key])       # pass imagepath along with class label
                        logger.debug("ReadWorker: Pushed item {} into File Queue...".format(key))
                        fileQueue.put(item)
                        imageNumber -= 1        # decrement for the thread to distinguish

            readFlag.value = 1

    def readNumeric(self, fileQueue, data_dir, options, readFlag):
        label = options['label']
        data_key = options['data_key']
        file = os.path.join(data_dir, os.listdir(data_dir)[0])
        if file.endswith('.csv'):
            try:
                df = pd.read_csv(file)
            except IOError as e:
                logger.error("Error reading csv file: ", exc_info=True)
        elif file.endswith('.json'):
            try:
                json_dict = json.load(open(file))
                cols = json_dict[data_key][0].keys()
                rows = []
                for row in json_dict[data_key]:
                    rows.append(row.values())

                df = pd.DataFrame(rows, columns=cols)
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

            item = tuple(['numeric', np_data, label, idx])
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
            item = tuple(['text', textDatum, textLabel, idx])
            logger.debug("Pushed item {} into File Queue...".format(idx + 1))
            fileQueue.put(item)

        readFlag.value = 1

class datumWorker():
    '''
    A worker for preparing the datum
    '''

    def __init__(self, fileQueue, datumQueue):
        count = 0
        while True:
            received = fileQueue.get()
            count += 1 
            dataType = received[0]
            received = received[1:].append(count)

            if dataType == 'image':
                self.ImageDatum(received, fileQueue, datumQueue)
            elif dataType == 'numeric':
                self.NumericDatum(received, fileQueue, datumQueue)
            elif dataType == 'text':
                self.TextDatum(received, fileQueue, datumQueue)
            else:
                logger.debug("Not the dataType I was expecting. Something broke.")
                sys.exit(-1)

    def ImageDatum(self, received, fileQueue, datumQueue):
        data, slabel, imageNumber, key, count = received
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
        data, label, key, count = received
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
        data, label, key, count = received
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
    def __init__(self, datumQueue, env, dbHandles):
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

                labelDBHandle = dbHandles[-1]
                with env.begin(write=True, db=labelDBHandle) as txn:
                    txn.put(str(key).encode('ascii'), label.SerializeToString())
            else:
                dbHandle = dbHandles[dbId - 1]
                with env.begin(write=True, db=dbHandle) as txn:
                    txn.put(str(key).encode('ascii'), datum.SerializeToString())

            logger.debug("WriteWorker: Datum #{} written to lmdb".format(key))


class Serialize():

    def __init__(self, nInputPerRecord=1, multi_input=False, nOutputPerRecord=1, multi_output=False):
        self.multi_input = bool(multi_input)
        self.multi_output = bool(multi_output)

        self.nInputPerRecord = int(nInputPerRecord)
        self.nOutputPerRecord = int(nOutputPerRecord)

        # config: 2 queues, 3 workers
        self.fileQueue = Queue()
        self.datumQueue = Queue()
        # because can have multiple input or output data to be read
        self.read_workers = []

        self.env = lmdb.open('lmdb/datumdb', max_dbs=(nInputPerRecord + nOutputPerRecord))

        if multi_input or multi_output:
            self.readFlags = manager.list([0]*(nInputPerRecord + nOutputPerRecord))     # create readFlags for each worker
        else:
            self.readFlags = manager.list([0]*nInputPerRecord)     # create readFlags for just one read_worker

        self.doneFlag = manager.Value('i', 0)
        logger.debug("Serialize Instance created with {} dbs".format(nInputPerRecord + nOutputPerRecord))

    def writeToLmdb(self, options):
        data_dir, args = options
        logger.debug("Data directory: " + str(data_dir))
        logger.info("Writing to LMDB")

        # save the dbnames for deserialization later
        self.nameddbs = []

        dbHandles = []
        # create dbs for datums
        for i in xrange(1,self.nInputPerRecord):
            dbName = 'datumdb' + str(i)
            self.nameddbs.append(dbName)
            dbHandles.append(self.env.open_db(dbName))
        # create labeldb
        for i in xrange(self.nOutputPerRecord):
            dbName = 'labeldb' + str(i)
            self.nameddbs.append(dbName)
            dbHandles.append(self.env.open_db(dbName))

        if not self.multi_input and not self.multi_output:
            '''
            single-input type, i.e. output should be inferred
            '''
            input_dict = args['input'][0]
            dataType = input_dict['dataType']

            if dataType == 'image':
                self.read_workers.append(Thread(target=readWorker().readImage, args=(self.fileQueue, data_dir, self.nInputPerRecord, self.readFlag)))

            elif dataType == 'numeric':
                self.read_workers.append(Thread(target=readWorker().readNumeric(self.fileQueue, data_dir, input_dict, self.readFlag)))

            elif dataType == 'text':
                self.read_workers.append(Thread(target=readWorker().readText, args=(self.fileQueue, data_dir, input_dict, self.readFlag)))

            else:
                logger.error(" :Incorrect or no tag given. Specify dataType in request.")
                sys.exit(-1)

        else:
            '''
            multi-input multi-output type
            Provide details for input and output streams.
            '''
            '''
            Image Bindings explained
            ========================
            File formats: .csv or .json

            .json types:
            ------------ 
            1. list of bindings(dicts)
                example:
                [
                    {
                        "file": "train_0.jpg",
                        "key": "0"
                    },
                    {
                        "file": "train_1.jpg",
                        "key": "1"
                    }
                ]

                Things you'll provide in the form: image_binding filename

            2. A dict with a key holding the list of bindings
                example:
                {
                    "bindings":
                    [
                        {
                            "file": "train_0.jpg",
                            "key": "0"
                        },
                        {
                            "file": "train_1.jpg",
                            "key": "1"
                        }
                    ]
                }

                Things you'll provide in the form: image_binding filename, data_key
            '''

            if 'image_binding' in args:
                image_binding_dict = args['image_binding']

            if image_binding_file.endswith('.csv'):
                logger.debug("found csv image binding")
                image_binding_df = pd.read_csv(image_binding_dict['file'])
            elif image_binding_file.endswith('.json'):
                logger.debug("found json image binding")
                parsed_json = json.load(image_binding_dict['file'])
                if isinstance(parsed_json, dict) and 'data_key' in image_binding_dict:      # second case
                    # just a sanity check: if it's a dict, data_key has to be provided
                    bindings_dict = parsed_json[image_binding_dict['data_key']]
                else:
                    bindings_dict = parsed_json

                cols = bindings_dict[0].keys()
                bindings = []
                for binding_dict in bindings_dict:
                    bindings.append(binding_dict.values())

                image_binding_df = pd.DataFrame(bindings, columns=cols)

            # for input_data in args['input']:
            #     dataType = input_data['dataType']


        self.datum_worker = Thread(target=datumWorker, args=(self.fileQueue, self.datumQueue,))
        self.write_worker = Thread(target=writeWorker, args=(self.datumQueue, self.env, dbHandles))

        for idx, worker in enumerate(self.read_workers):
            worker.setDaemon(True)
            worker.start()
            logger.debug("Read Worker #{} started".format(idx))
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
        dbHandles = []
        # create dbs for datums
        for i in xrange(1,nInputPerRecord):
            dbName = 'datumdb' + str(i)
            dbHandles.append(env.open_db(dbName))
        # create labeldb
        dbHandles.append(env.open_db('labeldb'))

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

        write_worker = Thread(target=writeWorker, args=(datumQueue, env, dbHandles))

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
