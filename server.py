from twisted.web.server import Site, NOT_DONE_YET
from twisted.web.resource import Resource
from twisted.internet import reactor, threads
import urllib
import os, zipfile, json
import logging
import urllib2, sys
# custom module
from serialize import Serialize
from src.models import train_model_2
print("New script imported.")

PORT = 8000

LMDB_DIR = 'data/interim/lmdb/traindb-jpg'
ZIPPED_FILE = 'datasets/dataset.zip'
DATA_DIR = 'data/processed'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not os.path.exists(LMDB_DIR):
    os.makedirs(LMDB_DIR)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


class DownloadFile(Resource):
    isLeaf = True
    def __init__(self):
        self.numberRequests = 0
        self.data = None
        self.serialized_flag = False

    def serverStart(self):
        logger.info("Server starting at port {}...\nPress Ctrl + C to stop.\n".format(PORT))
    
    '''
    Assumptions:
    For Status/http-form: GET request on the root
    For Serialization: POST request on the root
    For Training: GET request on some sub-domain
    '''

    def render_GET(self, request):
        logger.info("\nGET request received!")
        self.numberRequests += 1
        if self.data:
            logger.debug("Serialization in progress." + 
                "\nChecking whether data Serialization has completed.")
            self.joinThreads(request)
            return NOT_DONE_YET
        else:
            response = "Send a POST request to the same address to serialize the data.\n"
            return response

    def render_POST(self, request):
        logger.info("\nPOST request received!")
        req_dict = json.loads(request.content.getvalue())

        if 'already_serialized' in req_dict and req_dict['already_serialized'] == 'True':
            self.serialized_flag = True
        if self.serialized_flag is not True:
            if self.data is None:
                if req_dict['command'] == 'serialize':
                    logger.debug("id: " + req_dict['id'])
                    logger.debug("url: " + req_dict['url'])
                    logger.info("Fetching the dataset...")
                    reactor.callInThread(self.downloadFile, request)
                elif req_dict['command'] == 'deserialize':
                    logger.info("Whoops, Tried deserializing before serialization.")
                    return "Cannot deserialize before serialization."
                else:
                    logger.info("Unknown command passed.")
                    return "Please provide a valid command."
            else:
                logger.debug("Serialization in progress." + 
                    "\nChecking whether data Serialization has completed.")
                self.joinThreads(request)
                return NOT_DONE_YET

        else:
            if req_dict['command'] == 'deserialize':
                logger.info("Starting Deserialization and Training.")
                reactor.callInThread(self.deserialize, req_dict)
                return "Started training. Sit back."
            elif req_dict['command'] == 'serialize':
                logger.info("Tried serializing again.")
                return "Serialization already done. You can deserialize it now."
            else: 
                logger.info("Unknown command passed.")
                return "Please provide a valid command."
        return NOT_DONE_YET

    def downloadFile(self, request):
        args = json.loads(request.content.getvalue())
        url = args['url']
        filename = ZIPPED_FILE
        try:
            # urllib.urlretrieve(url, filename)
            # u = urllib2.urlopen(url)
            # h = u.info()
            # totalSize = int(h["Content-Length"])
            # if not totalSize:
            #     logger.error("Dataset not found.")

            # logger.debug("Downloading {} bytes...\n".format(totalSize))
            # fp = open(filename, 'wb')

            # blockSize = 8192    # urllib.urlretrieve uses 8192
            # count = 0
            # while True:
            #     chunk = u.read(blockSize)
            #     if not chunk: break
            #     fp.write(chunk)
            #     count += 1
            #     if totalSize > 0:
            #         percent = int(count * blockSize * 100 / totalSize)
            #         if percent > 100: percent = 100
            #         if percent < 100:
            #             sys.stdout.write("\r{}% downloaded".format(percent))
            #             sys.stdout.flush()
            #         else:
            #             sys.stdout.write("\nDone.")
            # fp.flush()
            # fp.close()

            logger.info("Download finished!")
            request.write("Dataset downloaded.")
            request.finish()
        except Exception as e:
            logger.error(e)
            request.write("Error downloading dataset.")
            request.finish()

        if len(args['input']) > 1:
            multi_input = True
            nInputPerRecord = len(args['input'])
        else:
            multi_input = False
            if 'nInputPerRecord' in args['input'][0]:
                # multi images per record in the same folder
                nInputPerRecord = args['input'][0]['nInputPerRecord']
            else: nInputPerRecord = 1

        if 'output' in args.keys():
            multi_output = True
            nOutputPerRecord = len(args['output'])
        else:
            multi_output = False
            nOutputPerRecord = 1

        self.data = Serialize()
        self.data._init_write(nInputPerRecord, multi_input, nOutputPerRecord, multi_output, lmdbPath=LMDB_DIR)
        self.d = threads.deferToThread(self.unzip, filename, args)
        self.d.addCallback(self.data.writeToLmdb)
        self.d.addErrback(self.errHandler)
        self.d.addErrback(self.errHandler)


    def errHandler(self, err):
        logger.error("Error caught in callback chain: ", exc_info=True)

    def unzip(self, filename, args):
        unzipped_dir = DATA_DIR
        try:
            # logger.info("Unzipping the file...")
            # with zipfile.ZipFile(filename, 'r') as zipref:
            #     zipref.extractall(unzipped_dir)
            # logger.info("Dataset extracted.")

            # os.remove(ZIPPED_FILE)    #get rid of the zip
            # logger.debug("got rid of the stupid zip")
            return list([unzipped_dir, args])

        except Exception as e:
            logger.error("Error extracting the zip: ", exc_info=True)
            sys.exit(-1)


    def joinThreads(self, request):
        logger.debug("inside joinThreads")
        logger.debug("readFlags: " + repr([flag.value for flag in self.data.readFlags]))
        
        if self.data.doneFlag.value == 1:
            if self.data.fileQueue.empty() and self.data.datumQueue.empty():
                logger.debug("Done with everything. Closing the lmdb environment.")
                self.data.env.close()
                logger.debug("Closed the lmdb environment.")
                logger.info("Data Serialization complete.")
                self.serialized_flag = True
                request.write("Data Serialization complete!.\n")
                request.finish()
            else:
                logger.debug("Waiting for queues to be empty.")
                request.write("Serializing the data. Try again later.\n")
                request.finish()

        elif all(flag.value == 1 for flag in self.data.readFlags):
            logger.debug("Reading complete. Joining read_worker.")
            for worker in self.data.read_workers:
                worker.join()
            self.data.doneFlag.value = 1
            logger.debug("Joined read_worker.")
            request.write("Serializing the data. Try again later.\n")
            request.finish()
        else:
            logger.debug("Reading not yet complete.")
            request.write("Serializing the data. Try again later.\n")
            request.finish()

    def deserialize(self, req_dict):
        self.data = Serialize() 
        return_dict = self.data.deserialize(req_dict)
        self.model = train_model_2.VGG19_mod()

        # build_dict = dict((k,return_dict[k]) for k in ['input_shapes', 'output_shapes'] if k in return_dict)

        self.model.build(return_dict)

        self.model.train(return_dict)

if __name__ == '__main__':
    root = Resource()
    downloadResource = DownloadFile()
    root.putChild("download", downloadResource)
    factory = Site(root)

    reactor.callWhenRunning(downloadResource.serverStart)
    reactor.listenTCP(PORT, factory)
    reactor.run()
