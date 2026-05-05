from threading import Event as ThreadEvent

class OrderTracker:
    """Track order status and cancellation."""
    def __init__(self, logger):
        self.order_accepted = ThreadEvent()
        self.order_canceled = ThreadEvent()
        self.order_id = None
        self.status = None
        self.logger = logger

    def on_logon(self, session_id, **kw):
        """Event handler for successful logon."""
        self.logger.info(f"FIX session established: {session_id}")


    def on_logout(self, session_id, reason=None, **kw):
        """Event handler for logout."""
        self.logger.info(f"FIX session closed: {session_id}, reason: {reason}")


    def on_reject(self, reason, msg_type, **kw):
        """Event handler for rejected messages."""
        self.logger.error(f"Message rejected - Type: {msg_type}, Reason: {reason}")
        
    def on_order_accepted(self, cl_ord_id, status, **kwargs):
        """Handle order accepted event."""
        if cl_ord_id == self.order_id:
            self.status = status
            self.logger.info(f"Order accepted: {cl_ord_id[:8]}... (Status: {status})")
            self.order_accepted.set()
    
    def on_order_canceled(self, orig_cl_ord_id, status, **kwargs):
        """Handle order canceled event."""
        if orig_cl_ord_id == self.order_id:
            self.status = status
            self.logger.info(f"Order canceled: {orig_cl_ord_id[:8]}... (Status: {status})")
            self.order_canceled.set()
    
    def on_order_rejected(self, cl_ord_id, reason, **kwargs):
        """Handle order rejected event."""
        if cl_ord_id == self.order_id:
            self.logger.error(f"Order rejected: {cl_ord_id[:8]}... - {reason}")