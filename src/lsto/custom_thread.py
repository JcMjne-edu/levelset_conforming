import threading

class CustomThread(threading.Thread):
  def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
    super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs, daemon=daemon)
    self._return=None

  def run(self):
    if self._target is not None:
      self._return=self._target(*self._args, **self._kwargs)

  def join(self):
    super().join()
    return self._return
    