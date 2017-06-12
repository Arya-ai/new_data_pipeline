from twisted.web.server import Site, NOT_DONE_YET
from twisted.web.resource import Resource
from twisted.internet import reactor, threads
import urllib
import os, zipfile, json
import logging
import urllib2, sys
# custom module
import serialize

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DownloadFile(Resource):
    isLeaf = True
    def __init__(self):
        self.numberRequests = 0
        self.serialize = None

    def serverStart(self):
        logger.info("Server starting...\nPress Ctrl + C to stop.\n")
    
    '''
    Assumptions:
    For Status/http-form: GET request on the root
    For Serialization: POST request on the root
    For Training: GET request on some sub-domain
    '''

    def render_GET(self, request):
        logger.info("\nGET request received!")
        self.numberRequests += 1
        if self.serialize:
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

        logger.debug("id: " + req_dict['id'])
        logger.debug("url: " + req_dict['url'])
        logger.info("Fetching the dataset...")
        
        reactor.callInThread(self.downloadFile, request)
        return NOT_DONE_YET

    # def print_request(self, request):
    #     req_dict = json.loads(request.content.getvalue())
    #     print req_dict
    #     request.write("request accepted!")
    #     request.finish()

    def downloadFile(self, request):
        args = json.loads(request.content.getvalue())
        url = args['url']
        filename = "dataset.zip"
        try:
            urllib.urlretrieve(url, filename)
            u = urllib2.urlopen(url)
            h = u.info()
            totalSize = int(h["Content-Length"])
            if not totalSize:
                logger.error("Dataset not found.")

            logger.debug("Downloading {} bytes...\n".format(totalSize))
            fp = open(filename, 'wb')

            blockSize = 8192    # urllib.urlretrieve uses 8192
            count = 0
            while True:
                chunk = u.read(blockSize)
                if not chunk: break
                fp.write(chunk)
                count += 1
                if totalSize > 0:
                    percent = int(count * blockSize * 100 / totalSize)
                    if percent > 100: percent = 100
                    if percent < 100:
                        sys.stdout.write("\r{}% downloaded".format(percent))
                        sys.stdout.flush()
                    else:
                        sys.stdout.write("\nDone.")
            fp.flush()
            fp.close()

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

        self.serialize = serialize.Serialize(nInputPerRecord, multi_input,nOutputPerRecord, multi_output)
        self.d = threads.deferToThread(self.unzip, filename, args)

        self.d.addCallback(self.serialize.writeToLmdb)
        self.d.addErrback(self.errHandler)
        self.d.addErrback(self.errHandler)


    def errHandler(self, err):
        logger.error("Error caught in callback chain: ", exc_info=True)

    def unzip(self, filename, args):
        unzipped_dir = "../datasets/dataset"
        try:
            logger.info("Unzipping the file...")
            with zipfile.ZipFile(filename, 'r') as zipref:
                zipref.extractall(unzipped_dir)
            logger.info("Dataset extracted.")

            os.remove("dataset.zip")    #get rid of the zip
            logger.debug("got rid of the stupid zip")
            return list([unzipped_dir, args])

        except Exception as e:
            logger.error("Error extracting the zip: ", exc_info=True)
            sys.exit(-1)


    def joinThreads(self, request):
        logger.debug("inside joinThreads")
        logger.debug("readFlags: " + repr([flag.value for flag in self.serialize.readFlags]))
        
        if self.serialize.doneFlag.value == 1:
            if self.serialize.fileQueue.empty() and self.serialize.datumQueue.empty():
                logger.debug("Done with everything. Closing the lmdb environment.")
                self.serialize.env.close()
                logger.debug("Closed the lmdb environment.")
                logger.info("Data Serialization complete.")
                request.write("Data Serialization complete!.\n")
                request.finish()
            else:
                logger.debug("Waiting for queues to be empty.")
                request.write("Serializing the data. Try again later.\n")
                request.finish()

        elif all(flag.value == 1 for flag in self.serialize.readFlags):
            logger.debug("Reading complete. Joining read_worker.")
            for worker in self.serialize.read_workers:
                worker.join()
            self.serialize.doneFlag.value = 1
            logger.debug("Joined read_worker.")
            request.write("Serializing the data. Try again later.\n")
            request.finish()
        else:
            logger.debug("Reading not yet complete.")
            request.write("Serializing the data. Try again later.\n")
            request.finish()


if __name__ == '__main__':
    root = Resource()
    downloadResource = DownloadFile()
    root.putChild("download", downloadResource)
    factory = Site(root)

    reactor.callWhenRunning(downloadResource.serverStart)
    reactor.listenTCP(8000, factory)
    reactor.run()
