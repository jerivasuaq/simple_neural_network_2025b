import numpy as np

# We'll use a corrected version of the data loading function from before
def get_mnist_corrected():
    import gzip, os, struct
    from urllib.request import urlretrieve
    def parse_data(src_url, num_samples, is_labels=False):
        gzfname, _ = urlretrieve(src_url, "./delete.me")
        try:
            with gzip.open(gzfname, 'rb') as gz:
                magic, num_items = struct.unpack('>II', gz.read(8))
                if is_labels: data = np.frombuffer(gz.read(), dtype=np.uint8)
                else: data = np.frombuffer(gz.read(), dtype=np.uint8).reshape(num_samples, 28, 28)
        finally: os.remove(gzfname)
        return data
    server = 'https://raw.githubusercontent.com/fgnt/mnist/master'
    x_train = parse_data(f'{server}/train-images-idx3-ubyte.gz', 60000)/255.
    y_train = parse_data(f'{server}/train-labels-idx1-ubyte.gz', 60000, is_labels=True)
    return x_train, y_train
