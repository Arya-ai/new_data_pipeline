import lmdb
import numpy as np
import cv2  #openCV for getting image attributes
import tifffile as tiff
import os, sys
from subprocess import Popen, PIPE
from re import search
from threading import Thread
from datum_pb2 import Datum
import pandas as pd
from multiprocessing import Manager
from Queue import Queue
from sklearn.feature_extraction.text import CountVectorizer
import json
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

manager = Manager()

class readWorker():
    '''
    A worker for reading the data
    '''
    def readImage(self, fileQueue, data_dir, readFlag, nInputPerRecord=None, dataFlow=None, dbId=None, binding_df=None, options=None):
        logger.debug("\nImage dir: " + data_dir)
        if dataFlow is not None and binding_df is not None:
            # multi-input/ multi-output case
            key = 0
            for record in tqdm(binding_df):
                key += 1
                imagePath = os.path.join(data_dir, str(record)) + str(options['extension'])
                # Open CV does not easily support tiff images
                if imagePath.endswith('.tif'):
                    ndarray = tiff.imread(imagePath)
                else:
                    ndarray = cv2.imread(imagePath)
                task_dict = {'data': ndarray, 'dataType': 'image', 'key': key, 'dbId': dbId, 'dataFlow': dataFlow}
                logger.debug("ImageReader: Pushed item {} into File Queue...".format(key))
                fileQueue.put(task_dict)

            readFlag.value = 1
            logger.debug("ImageReader: Image ReadFlag #{} set.".format(task_dict['dbId']))

        else:
            # single input case
            if nInputPerRecord == 1:
                '''
                Directly read the images from each subdirectory (label directory)
                '''
                imageNumber = 0
                labeldirs = [os.path.join(data_dir, subdir) for subdir in os.listdir(data_dir)]

                key = 0

                for labeldir in labeldirs:
                    for image in tqdm(os.listdir(labeldir)):
                        key += 1
                        imagePath = os.path.join(labeldir, image)
                        ndarray = cv2.imread(imagePath)
                        slabel = imagePath.split('/')[-2]
                        task_dict = {'data': ndarray, 'dataType': 'image', 'label': slabel, 'key': key, 'dbId': 0}
                        logger.debug("ImageReader: Pushed item {} into File Queue...".format(key))
                        fileQueue.put(task_dict)     # pass imagepath along with class label

                readFlag.value = 1
                logger.debug("ImageReader: Image ReadFlag #{} set.".format(task_dict['dbId']))

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

                    imageNumber = nInputPerRecord - 1
                    # goes from (n - 1) to 0
                    for image in images:
                        key += 1
                        for im, rootPath in tqdm(zip(image, label_list)):
                            imagePath = os.path.join(rootPath, im)

                            try:
                                ndarray = cv2.imread(imagePath)
                            except IOError as e:
                                logger.error("Error:", e)
                            slabel = imagePath.split('/')[-2]
                            task_dict = {'data': ndarray, 'dataType': 'image', 'label':slabel, 'key': key, 
                                        'dbId': imageNumber, 'multiImage': True}
                            # special case: Multi inputs but output label is inferred, not supplied separately
                            logger.debug("ImageReader: Pushed item {} into File Queue...".format(key))
                            fileQueue.put(task_dict)
                            imageNumber -= 1        # decrement for the thread to distinguish

                readFlag.value = 1
                logger.debug("ImageReader: Image ReadFlag #{} set.".format(task_dict['dbId']))


    def readNumeric(self, fileQueue, file, readFlag, options=None, dataFlow=None, dbId=None):
        logger.debug("\nNumeric filepath: " + file)
        if file.endswith('.csv'):
            try:
                df = pd.read_csv(file)
            except IOError as e:
                logger.error("Error reading csv file: ", exc_info=True)
                sys.exit(-1)
        elif file.endswith('.json'):
            try:
                parsed_json = json.load(open(file))
                if isinstance(parsed_json, dict) and 'data_key' in options.keys():
                    # just a sanity check: if it's a dict, data_key has to be provided
                    datum_dicts = parsed_json[options['data_key']]
                else:
                    datum_dicts = parsed_json

                cols = datum_dicts[0].keys()
                rows = []
                for datum_dict in datum_dicts:
                    rows.append(datum_dict.values())

                df = pd.DataFrame(rows, columns=cols)
            except IOError as e:
                logger.error("Error opening json file: ", exc_info=True)
                sys.exit(-1)
        else:
            logger.error("Error: Provide the file in valid format (.csv or .json)")
            sys.exit(-1)

        if options and 'label' in options:
            label = options['label']
            # labelSeries = []
            # for label in labels:
            #     labelSeries.append(df.pop(label))

            # labeldf = pd.concat(labelSeries, axis=1)        # concat all the labels into a single df
            labeldf = df.pop(label)
        else: labeldf = None

        for idx in tqdm(xrange(len(df))):
            data = df.iloc[idx].to_frame().to_records(index=False)
            '''
            this will return a numpy record array which needs to be converted to a numpy array
            for it to be converted back from a byte array
            '''
            # retrieve the datatype
            dt = data.dtype[0]
            # convert into numpy array
            np_data = np.array(data.view(dt))

            task_dict = {'data': np_data, 'dataType': 'numeric', 'key': idx + 1}

            if labeldf:
                label_data = labeldf.iloc[idx]       # for single label
                # label_data = label_data.to_frame()
                # label_data = label_data.to_records(index=False)
                task_dict['label'] = label_data

            if dbId:
                task_dict['dbId'] = dbId        # MIMO
            else: task_dict['dbId'] = 0

            if dataFlow:
                task_dict['dataFlow'] = dataFlow

            logger.debug("NumericReader: Pushed item {} into File Queue...".format(idx + 1))
            fileQueue.put(task_dict)

        readFlag.value = 1
        logger.debug("NumericReader: Numeric ReadFlag #{} set.".format(task_dict['dbId']))


    def readText(self, fileQueue, file, readFlag, options=None, dataFlow=None, dbId=None):
        logger.debug("\nText filepath: " + file)

        if file.endswith(".csv"):
            try:
                df = pd.read_csv(file)
            except IOError as e:
                logger.error("Error:", e)
                sys.exit(-1)
        elif file.endswith(".json"):
            try:
                parsed_json = json.load(open(file))
                if isinstance(parsed_json, dict) and 'data_key' in options.keys():
                    # just a sanity check: if it's a dict, data_key has to be provided
                    datum_dicts = parsed_json[options['data_key']]
                else:
                    datum_dicts = parsed_json

                cols = datum_dicts[0].keys()
                rows = []
                for datum_dict in datum_dicts:
                    rows.append(datum_dict.values())

                df = pd.DataFrame(rows, columns=cols)
            except IOError as e:
                logger.error("Error opening json file: ", exc_info=True)
                sys.exit(-1)
        else:
            logger.error("Error: Provide the file in valid format (.csv or .json)")
            sys.exit(-1)

        vectorizer = CountVectorizer(token_pattern=u"(?u)\\b\\w+\\b")

        if options and 'text' in options.keys(): text_field = options['text']
        else: text_field = df.columns[0]
        # in case of a list of strings (text)

        caps = []
        for text_datum in df[text_field]:
            caps.append(text_datum)

        train_features = vectorizer.fit_transform(caps)
        vecs = train_features.toarray()

        for idx in tqdm(xrange(len(df[name]))):
            textDatum = vecs[idx]
            task_dict = {'data': text_datum, 'dataType': 'text', 'key': idx + 1}
            if options and 'label' in options.keys():
                textLabel = df[name][idx][label]
                task_dict['label'] = textLabel

            logger.debug("TextReader: Pushed item {} into File Queue...".format(idx + 1))
            fileQueue.put(task_dict)

        readFlag.value = 1
        logger.debug("TextReader: Text ReadFlag #{} set.".format(task_dict['dbId']))


class datumWorker():
    '''
    A worker for preparing the datum
    '''

    def __init__(self, fileQueue, datumQueue):
        self.count = 0
        while True:
            task_dict = fileQueue.get()
            self.count += 1 
            dataType = task_dict.pop('dataType')

            if dataType == 'image':
                self.ImageDatum(task_dict, fileQueue, datumQueue)
            elif dataType == 'numeric':
                self.NumericDatum(task_dict, fileQueue, datumQueue)
            elif dataType == 'text':
                self.TextDatum(task_dict, fileQueue, datumQueue)
            else:
                logger.debug("Not the dataType I was expecting. Something broke.")
                sys.exit(-1)

    def ImageDatum(self, task_dict, fileQueue, datumQueue):
        datum = Datum()

        dims = task_dict['data'].shape

        imageDatum = datum.imgdata
        imageDatum.identifier = str(task_dict['key'])
        imageDatum.channels = dims[2]
        imageDatum.height = dims[0]
        imageDatum.width = dims[1]
        data = task_dict.pop('data')
        data = np.asarray(data, dtype='float32')
        imageDatum.data = data.tobytes()

        task_dict['datum'] = datum
    
        if 'label' in task_dict.keys():
            labelDatum = datum.classs
            labelDatum.identifier = str(task_dict['key'])
            labelDatum.slabel = task_dict.pop('label')

            task_dict['label'] = datum

        datumQueue.put(task_dict)
        logger.debug("ImageDatum: Pushed datum #{} into Datum Queue".format(self.count))
        fileQueue.task_done()       # let the fileQueue know item has been processed and is safe to delete

    def NumericDatum(self, task_dict, fileQueue, datumQueue):
        datum = Datum()

        numericDatum = datum.numeric
        numericDatum.identifier = str(task_dict['key'])
        # numericDatum.size.dim = label.shape[0]
        numericDatum.size.dim = 1       # currently just for 1 label
        data = task_dict.pop('data')
        data = np.asarray(data, dtype='float32')
        numericDatum.data = data.tobytes()

        task_dict['datum'] = datum

        if 'label' in task_dict.keys():
            labelDatum = datum.classs
            labelDatum.identifier = str(task_dict['key'])
            labelDatum.nlabel = task_dict.pop('label')
            task_dict['label'] = datum

        datumQueue.put(task_dict)
        logger.debug("NumericDatum: Pushed datum #{} into Datum Queue".format(self.count))
        fileQueue.task_done()

    def TextDatum(self, fileQueue, datumQueue):
        datum = Datum()

        textDatum = datum.numeric
        textDatum.identifier = str(task_dict['key'])
        textDatum.size.dim = 1
        data = task_dict.pop('data')
        data = np.asarray(data, dtype='float32')
        textDatum.data = data.tobytes()

        task_dict['datum'] = datum

        if 'label' in task_dict.keys():
            labelDatum = datum.classs
            labelDatum.identifier = str(task_dict['key'])
            labelDatum.nlabel = task_dict.pop('label')

            task_dict['label'] = datum

        datumQueue.put(task_dict)
        logger.debug("TextDatum: Pushed datum #{} into Datum Queue".format(self.count))
        fileQueue.task_done()
            
class writeWorker():
    '''
    A worker for writing the datum to lmdb database
    '''
    def __init__(self, datumQueue, env, inputDBHandles, outputDBHandles):
        while True:
            task_dict = datumQueue.get()

            if 'multiImage' in task_dict.keys() and task_dict['multiImage']:
                # special case with the images. one label and multilpe images.
                # Refer definition of readWorker.readImage
                if task_dict['dbId'] == 0:
                    datumDBHandle = inputDBHandles[task_dict['dbId']]
                    with env.begin(write=True, db=datumDBHandle) as txn:
                        txn.put(str(task_dict['key']).encode('ascii'), task_dict['datum'].SerializeToString())

                    labelDBHandle = outputDBHandles[-1]
                    with env.begin(write=True, db=labelDBHandle) as txn:
                        txn.put(str(task_dict['key']).encode('ascii'), task_dict['label'].SerializeToString())
                else:
                    datumDBHandle = inputDBHandles[task_dict['dbId']]
                    with env.begin(write=True, db=datumDBHandle) as txn:
                        txn.put(str(task_dict['key']).encode('ascii'), task_dict['datum'].SerializeToString())

            elif 'dataFlow' in task_dict.keys():
                # MIMO - no label, just datum
                if task_dict['dataFlow'] == 'input':
                    DBHandle = inputDBHandles[task_dict['dbId']]
                elif task_dict['dataFlow'] == 'output':
                    DBHandle = outputDBHandles[task_dict['dbId']]
                with env.begin(write=True, db=DBHandle) as txn:
                    txn.put(str(task_dict['key']).encode('ascii'), task_dict['datum'].SerializeToString())

            else:
                # single-input
                datumDBHandle = inputDBHandles[task_dict['dbId']]
                with env.begin(write=True, db=datumDBHandle) as txn:
                    txn.put(str(task_dict['key']).encode('ascii'), task_dict['datum'].SerializeToString())

                if 'label' in task_dict.keys():
                    labelDBHandle = outputDBHandles['dbId']
                    with env.begin(write=True, db=labelDBHandle) as txn:
                        txn.put(str(task_dict['key']).encode('ascii'), task_dict['label'].SerializeToString())

            datumQueue.task_done()
            logger.debug("WriteWorker: Datum #{} written to lmdb".format(task_dict['key']))

class Serialize():

    def __init__(self):
        self.lmdbPath = None
        self.nInputPerRecord = None
        self.nOutputPerRecord = None

    def _init_write(self, nInputPerRecord=1, multi_input=False, nOutputPerRecord=1, multi_output=False, lmdbPath=None):
        self.multi_input = bool(multi_input)
        self.multi_output = bool(multi_output)

        self.nInputPerRecord = int(nInputPerRecord)
        self.nOutputPerRecord = int(nOutputPerRecord)

        # config: 2 queues, 3 workers
        self.fileQueue = Queue()
        self.datumQueue = Queue()
        # because can have multiple input or output data to be read
        self.read_workers = []

        if lmdbPath:
            self.lmdbPath = lmdbPath
        else:
            self.lmdbPath = 'lmdb/datumdb'

        if not os.path.exists(lmdbPath):
            os.makedirs(lmdbPath)

        if multi_input or multi_output:
            # create self.readFlags for each worker
            self.readFlags = [manager.Value('i', 0) for i in range(self.nInputPerRecord + self.nOutputPerRecord)]
        else:
            # create self.readFlags for just one read_worker
            self.readFlags = [manager.Value('i', 0)]

        self.doneFlag = manager.Value('i', 0)
        logger.debug("Read Flags: " + repr([flag.value for flag in self.readFlags]))
        logger.debug("Serialize Instance created with {} dbs".format(self.nInputPerRecord + self.nOutputPerRecord))


    def writeToLmdb(self, options):
        data_dir, args = options
        logger.debug("Data directory: " + str(data_dir))
        logger.info("Writing to LMDB")

        # The default mapsize for LMDB is 1 GB, and it does not have dynamic sizing.
        # So for larger datasets, the mapsize should be explicitly set to almost 
        # the size of the folder containing the dataset.

        proc = Popen(['du', '-s', data_dir], stdout=PIPE)
        shell_output = proc.communicate()[0]    # 0 = stdout, 1 = stderr
        mapsize = int(search(r'(\d+)', shell_output).group())

        self.env = lmdb.open(self.lmdbPath, max_dbs=(self.nInputPerRecord + self.nOutputPerRecord + 1), map_size=mapsize*1024*100)

        # if mapsize <= 1048576:
        #     self.env = lmdb.open(self.lmdbPath, max_dbs=(self.nInputPerRecord + self.nOutputPerRecord + 1), map_size=1048576*20)
        # else:
        #     self.env = lmdb.open(self.lmdbPath, max_dbs=(self.nInputPerRecord + self.nOutputPerRecord + 1), map_size=mapsize)

        '''
        why +1 than required?
        A: The main db stores the keys to all the named dbs, which are datumdbs and labeldbs.
        It does not contain any data to make the process more easily understandable; i.e. data in all the named dbs.
        '''

        # save the dbnames for deserialization later
        self.inputDBs = []
        self.outputDBs = []

        # get DB handles to pass along with the env
        inputDBHandles = []
        outputDBHandles = []
        # create dbs for datums
        for i in xrange(self.nInputPerRecord):
            dbName = 'datumdb' + str(i)
            self.inputDBs.append(dbName)
            inputDBHandles.append(self.env.open_db(dbName))
        # create labeldb
        for i in xrange(self.nOutputPerRecord):
            dbName = 'labeldb' + str(i)
            self.outputDBs.append(dbName)
            outputDBHandles.append(self.env.open_db(dbName))

        if not self.multi_input and not self.multi_output:
            '''
            single-input type, i.e. output should be inferred
            '''
            input_dict = args['input'][0]
            dataType = input_dict['dataType']

            if dataType == 'image':
                self.read_workers.append(Thread(target=readWorker().readImage(
                    self.fileQueue, data_dir, nInputPerRecord=self.nInputPerRecord, readFlag=self.readFlags[0])))

            elif dataType == 'numeric':
                file = os.path.join(data_dir, os.listdir(data_dir)[0])
                self.read_workers.append(Thread(target=readWorker().readNumeric(
                    self.fileQueue, file, readFlag=self.readFlags[0], options=input_dict)))

            elif dataType == 'text':
                file = os.path.join(data_dir, os.listdir(data_dir)[0])
                self.read_workers.append(Thread(target=readWorker().readText(
                    self.fileQueue, data_dir, readFlag=self.readFlags[0], options=input_dict)))

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
                image_binding_file = os.path.join(data_dir, image_binding_dict['file'])

            try:
                logger.debug("Reading image_binding file...")
                if image_binding_file.endswith('.csv'):
                    logger.debug("found csv image binding")
                    image_binding_df = pd.read_csv(image_binding_file)
                elif image_binding_file.endswith('.json'):
                    logger.debug("found json image binding")
                    parsed_json = json.load(image_binding_file)
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
            except Exception as e:
                logger.error("Error reading image binding file: " + str(e))

            for idx, input_data in enumerate(args['input']):
                dataType = input_data['dataType']

                if dataType == 'image':
                    directory = os.path.join(data_dir, input_data['directory'])
                    binding_field = input_data['binding_field']
                    binding_df = image_binding_df.pop(binding_field)
                    self.read_workers.append(Thread(target=readWorker().readImage(
                        self.fileQueue, directory, readFlag=self.readFlags[idx], 
                        dataFlow='input', dbId=idx, binding_df=binding_df, options=input_data)))

                elif dataType == 'numeric':
                    file = os.path.join(data_dir, input_data['file'])
                    self.read_workers.append(Thread(readWorker().readNumeric(
                        self.fileQueue, file, readFlag=self.readFlags[idx], dataFlow='input', dbId=idx)))

                elif dataType == 'text':
                    file = os.path.join(data_dir, input_data['file'])
                    self.read_workers.append(Thread(read_workers().readText(
                        self.fileQueue, file, options=input_dict, readFlag=self.readFlags[idx], dataFlow='input', dbId=idx)))

                else:
                    logger.error("Error reading data: invalid format provided.")
                    sys.exit(-1)


            for idx, output_data in enumerate(args['output']):
                dataType = output_data['dataType']

                if dataType == 'image':
                    directory = os.path.join(data_dir, output_data['directory'])
                    binding_field = output_data['binding_field']
                    binding_df = image_binding_df.pop(binding_field)
                    self.read_workers.append(Thread(target=readWorker().readImage(
                        self.fileQueue, directory, readFlag=self.readFlags[self.nInputPerRecord + idx], 
                        dataFlow='output', dbId=idx, binding_df=binding_df, options=output_data)))

                elif dataType == 'numeric':
                    file = os.path.join(data_dir, output_data['file'])
                    self.read_workers.append(Thread(readWorker().readNumeric(
                        self.fileQueue, file, readFlag=self.readFlags[self.nInputPerRecord + idx], dataFlow='output', dbId=idx)))

                elif dataType == 'text':
                    file = os.path.join(data_dir, output_data['file'])
                    self.read_workers.append(Thread(read_workers().readText(
                        self.fileQueue, file, readFlag=self.readFlags[self.nInputPerRecord + idx], dataFlow='output', dbId=idx)))

                else:
                    logger.error("Error reading data: invalid format provided.")
                    sys.exit(-1)

        self.datum_worker = Thread(target=datumWorker, args=(self.fileQueue, self.datumQueue,))
        self.write_worker = Thread(target=writeWorker, args=(self.datumQueue, self.env, inputDBHandles, outputDBHandles))

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


    def deserialize(self, options):

        self.lmdbPath = options['lmdbPath']
        self.batch_size = int(options['batch_size'])

        self.env = lmdb.open(self.lmdbPath, max_dbs=(self.nInputPerRecord + self.nOutputPerRecord + 1), readonly=True)

        inputDBHandles = []
        outputDBHandles = []
        # get dbHandles for inputs
        for idx in xrange(self.nInputPerRecord):
            dbName = self.inputDBs[idx]
            inputDBHandles.append(self.env.open_db(dbName))
        # get dbHandles for outputs
        for idx in xrange(self.nOutputPerRecord):
            dbName = self.outputDBs[idx]
            outputDBHandles.append(self.env.open_db(dbName))

        txn_nameless = self.env.begin(write=False)
        # input + output streams
        n_streams = txn_nameless.stat()['entries']
        logger.debug("Number of streams of data:" + repr(n_streams))

        try:
            assert int(n_streams) == self.nInputPerRecord + self.nOutputPerRecord
        except AssertionError:
            logger.error("Number of databases does not match with the given data.")

        with self.env.begin(write=False, db=inputDBHandles[0]) as txn_named:
            self.n_samples = txn_named.stat()['entries']
            logger.debug("Number of samples: " + repr(self.n_samples))

        # set number of batches
        self.n_batches = self.n_samples // self.batch_size

        input_cursors = [txn_nameless.cursor(inputDBHandle) for inputDBHandle in inputDBHandles]
        output_cursors = [txn_nameless.cursor(outputDBHandle) for outputDBHandle in outputDBHandles]

        logger.debug("Input cursors: " + repr(len(input_cursors)))
        logger.debug("Output cursors: " + repr(len(output_cursors)))

        input_shapes, output_shapes = self._get_datum_shapes(input_cursors, output_cursors)

        return_dict = {'input_shapes': input_shapes, 'output_shapes': output_shapes, 
                        'generator': self.batch_generator(input_cursors, output_cursors), 
                        'n_samples': self.n_samples, 'batch_size': self.batch_size}

        return return_dict


    def _get_datum_shapes(self, input_cursors, output_cursors):
        input_shapes = []
        output_shapes = []

        for input_cursor in input_cursors:
            input_cursor.first()
            _, raw_datum = input_cursor.item()
            datum = Datum()
            datum.ParseFromString(raw_datum)

            if datum.imgdata.data:
                flat_x = np.fromstring(datum.imgdata.data, dtype='float32')
                x = flat_x.reshape((datum.imgdata.height, datum.imgdata.width, datum.imgdata.channels))
                shape = (datum.imgdata.height, datum.imgdata.width, datum.imgdata.channels)
                logger.debug("input image shape: " + repr(shape))
                input_shapes.append(shape)

            if datum.numeric.data:
                flat_x = np.fromstring(datum.numeric.data, dtype='float32')
                shape = flat_x.shape
                logger.debug("input numeric shape: " + repr(shape))
                input_shapes.append(shape)

        for output_cursor in output_cursors:
            output_cursor.first()
            _, raw_datum = output_cursor.item()
            datum = Datum()
            datum.ParseFromString(raw_datum)

            if datum.imgdata.data:
                flat_x = np.fromstring(datum.imgdata.data, dtype='float32')
                x = flat_x.reshape((datum.imgdata.height, datum.imgdata.width, datum.imgdata.channels))
                shape = (datum.imgdata.height, datum.imgdata.width, datum.imgdata.channels)
                logger.debug("output image shape: " + repr(shape))
                output_shapes.append(shape)

            if datum.numeric.data:
                flat_x = np.fromstring(datum.numeric.data, dtype='float32')
                shape = flat_x.shape
                logger.debug("output numeric shape: " + repr(shape))
                output_shapes.append(shape)

            return (input_shapes, output_shapes)


    def batch_generator(self, input_cursors, output_cursors):
        while 1:    
            for idx in xrange(self.n_batches):
                inputs = []
                outputs = []

                for input_cursor in input_cursors:
                    n = self.batch_size
                    input_cursor.set_range(str(idx*self.batch_size + 1))
                    logger.debug("Current input key: " + repr(input_cursor.key()))
                    batch_inputs = []

                    for _, raw_datum in input_cursor:
                        if n <= 0: break

                        datum = Datum()
                        datum.ParseFromString(raw_datum)

                        if datum.imgdata.data:
                            flat_x = np.fromstring(datum.imgdata.data, dtype='float32')
                            x = flat_x.reshape((datum.imgdata.height, datum.imgdata.width, datum.imgdata.channels))
                            shape = (datum.imgdata.height, datum.imgdata.width, datum.imgdata.channels)
                            logger.debug("input image shape: " + repr(shape))
                            batch_inputs.append(x)
                        elif datum.numeric.data:
                            flat_x = np.fromstring(datum.numeric.data, dtype='float32')
                            logger.debug("input numeric shape: " + repr(flat_x.shape))
                            batch_inputs.append(flat_x)

                        n -= 1

                    batch_inputs = np.array(batch_inputs)
                    logger.debug("Batch input shape: " + repr(batch_inputs.shape))
                    inputs.append(batch_inputs)

                for output_cursor in output_cursors:
                    n = self.batch_size
                    output_cursor.set_range(str(idx*self.batch_size + 1))
                    logger.debug("Current output key: " + repr(output_cursor.key()))
                    batch_outputs = []

                    for _, raw_datum in output_cursor:
                        if n <= 0: break

                        datum = Datum()
                        datum.ParseFromString(raw_datum)

                        if datum.imgdata.data:
                            flat_x = np.fromstring(datum.imgdata.data, dtype='float32')
                            x = flat_x.reshape((datum.imgdata.height, datum.imgdata.width, datum.imgdata.channels))
                            shape = (datum.imgdata.height, datum.imgdata.width, datum.imgdata.channels)
                            logger.debug("output image shape: " + repr(shape))
                            batch_outputs.append(x)
                        elif datum.numeric.data:
                            flat_x = np.fromstring(datum.numeric.data, dtype='float32')
                            logger.debug("output numeric shape: " + repr(flat_x.shape))
                            batch_outputs.append(flat_x)

                        n -= 1

                    batch_outputs = np.array(batch_outputs)
                    logger.debug("Batch output shape: " + repr(batch_outputs.shape))
                    outputs.append(batch_outputs)

                yield(inputs, outputs)

