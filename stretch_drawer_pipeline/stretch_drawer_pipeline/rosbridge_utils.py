"""Rosbridge utilities for stretch_drawer_pipeline.

Provides ThreadedService — a roslibpy.Service subclass whose advertise
callback runs in a thread pool instead of on the twisted reactor.  This
lets the handler block (e.g. for Detic inference or TF lookups) without
freezing the roslibpy event loop, so image and TF callbacks keep firing.
The rosbridge service response is sent only after the handler returns.
"""

import roslibpy
from roslibpy import Message, ServiceResponse


class ThreadedService(roslibpy.Service):
    """Service whose advertise callback runs off the twisted reactor thread.

    Usage is identical to ``roslibpy.Service`` — the only difference is that
    the callback passed to ``advertise()`` executes on a thread-pool worker,
    so it may block for as long as needed.
    """

    def _service_response_handler(self, request):
        from twisted.internet import reactor

        def _run():
            response = ServiceResponse()
            success = self._service_callback(request["args"], response)
            call = Message({
                "op": "service_response",
                "service": self.name,
                "values": dict(response),
                "result": success,
            })
            if "id" in request:
                call["id"] = request["id"]
            self.ros.send_on_ready(call)

        reactor.callFromThread(reactor.callInThread, _run)