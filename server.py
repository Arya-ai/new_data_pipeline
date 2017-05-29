from twisted.web.server import Site, NOT_DONE_YET
from twisted.web.resource import Resource
from twisted.internet import reactor, threads
import urllib
import os, zipfile
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
    
    def render_GET(self, request):
        logger.info("\nGET request received!")
        self.numberRequests += 1
        response = ""
        if self.serialize:
            logger.debug("Checking whether data Serialization has completed")
            self.joinThreads(request)
            return NOT_DONE_YET
        else:
            response = "Send a POST request to the same address to serialize the data.\n"
            return response

    def render_POST(self, request):
        logger.info("\nPOST request received!")
        logger.debug("id: " + request.args['id'][0])
        logger.debug("url: " + request.args['url'][0])
        logger.info("Fetching the dataset...")
        reactor.callInThread(self.downloadFile, request)
        return NOT_DONE_YET

    def downloadFile(self, request):
        args = request.args
        url = args['url'][0]
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

        nInputPerRecord = None
        if 'nInputRecord' in args:
            nInputPerRecord = int(args['nInputPerRecord'][0])
        else: nInputPerRecord = 1

        self.serialize = serialize.Serialize(nInputPerRecord)
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

        elif self.serialize.readFlag.value == 1:
            logger.debug("Reading complete. Joining read_worker.")
            self.serialize.read_worker.join()
            self.serialize.doneFlag.value = 1
            logger.debug("Joined read_worker.")
            request.write("Serializing the data. Try again later.\n")
            request.finish()
        else:
            logger.debug("Reading not yet complete yet.")
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
