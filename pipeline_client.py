'''
A command line tool for sending POST or GET requests to the twisted server
for serialization/deserialization of datasets.

Usage: python pipeline_client.py --method [POST | GET] json_request_file
'''

from __future__ import print_function
import requests
import sys
import json

def send_request(method, requestFile=None):
	if method == 'POST' and requestFile:
		req_dict = json.load(open(requestFile))
		res = requests.post('http://localhost:8000/download', json=req_dict)
	else:
		res = requests.get('http://localhost:8000/download')

	print("Response:\n" + res.text)

if __name__ == '__main__':
	# discard the script name
	argv = sys.argv[1:]
	argv.reverse()

	if len(argv) < 2:
		print("Error: Not all parameters provided.")
		print("Usage: python pipeline_client.py --method [POST | GET] json_request_file")
		sys.exit(-1)

	if argv.pop() == '--method':
		method = str(argv.pop()).strip()
	else:
		print("Provide a method for the HTTP request.")
		sys.exit(-1)

	if method == 'POST':
		try:
			requestFile = argv.pop()
			send_request(method, requestFile)
		except IndexError as e:
			print("No request file provided for POST request.")
			print("Usage: python pipeline_client.py --method [POST | GET] json_request_file")
			sys.exit(-1)
	else:
		send_request(method)

# !end