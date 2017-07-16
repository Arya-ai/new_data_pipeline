
from __future__ import print_function

from twisted.internet import reactor, protocol


class Echo(protocol.Protocol):
    """This is just about the simplest possible protocol"""
    
    def serverStart(self):
        self.transport.write("server running")
        print("server running")

    def dataReceived(self, data):
        "As soon as any data is received, write it back."
        self.transport.write(data)
        print("Data received: " + data)


def main():
    """This runs the protocol on port 8000"""
    factory = protocol.ServerFactory()
    factory.protocol = Echo
    reactor.listenTCP(2048,factory)
    reactor.run()
    # print "Server running.."

# this only runs if the module was *not* imported
if __name__ == '__main__':
    main()

