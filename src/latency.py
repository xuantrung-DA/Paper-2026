# latency measurement helpers

def measure(func, *args, **kwargs):
    import time
    t0 = time.time()
    func(*args, **kwargs)
    return time.time() - t0
