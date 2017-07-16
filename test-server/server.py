from __future__ import absolute_import
from __future__ import print_function

from twisted.web.server import Site, NOT_DONE_YET
from twisted.web.resource import Resource
from twisted.internet import reactor, threads
import urllib
import os, zipfile, json
import logging
import urllib2, sys
from utils import get_image_from_buffer, predict, get_json_from_ndarray, loadModel, split
import numpy as np

PORT = 2048

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DownloadFile(Resource):
	isLeaf = True
	def __init__(self):
		self.numberRequests = 0
		self.model = None

	def serverStart(self):
		logger.info("Server starting on port {}...\nPress Ctrl + C to stop.\n".format(PORT))
	
	'''
	Assumptions:
	For Status/http-form: GET request on the root
	For Serialization: POST request on the root
	For Training: GET request on some sub-domain
	'''

	def render_GET(self, request):
		logger.info("\nGET request received!")
		self.numberRequests += 1
		response = "Send a POST request to the same address.\n"
		return response

	def render_POST(self, request):
		self.numberRequests += 1
		logger.info("\nPOST request received!")
		req_dict = json.loads(request.content.read())

		image_dict = req_dict['image']
		image_buffer = image_dict['data']
		img = get_image_from_buffer(image_buffer)

		if img is not None:
			if not self.model:
				self.model = loadModel()

			if img.shape[0] > 256 or img.shape[1] > 256:
				images = split(img)
				print("multi images:", type(images))
				confidence_ndarray = predict(self.model, images)
			else:
				confidence_ndarray = predict(self.model, img)

			print("Confidence:", confidence_ndarray)

			# if confidence_ndarray.shape[0] > 1:
			# 	confidence_ndarray = np.max(confidence_ndarray, axis=0)

			if confidence_ndarray is not None:
				status = True
				json_weather, json_landCover = get_json_from_ndarray(confidence_ndarray)
				response_dict = {'uid': req_dict['uid'], 'result': {'weather': json_weather, 'land_cover': json_landCover}, 'status': status}
			else:
				status = False
				result = None
				response_dict = {'uid': req_dict['uid'], 'result': None, 'status': status}

		else:
			status = False
			result = None
			response_dict = {'uid': req_dict['uid'], 'result': None, 'status': status}

		json_resp = json.dumps(response_dict)
		print("Response sent:", response_dict)

		request.responseHeaders.addRawHeader(b"content-type", b"application/json")
		return json_resp


if __name__ == '__main__':
	root = Resource()
	downloadResource = DownloadFile()
	root.putChild("satelliteImagery", downloadResource)
	factory = Site(root)

	reactor.callWhenRunning(downloadResource.serverStart)
	reactor.listenTCP(PORT, factory)
	reactor.run()
