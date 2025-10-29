import gzip
import numpy as np
import os
import struct
from urllib.request import urlretrieve
import matplotlib.pyplot as plt

def get_mnist():
    # The code to download the mnist data original came from
    # https://cntk.ai/pythondocs/CNTK_103A_MNIST_DataLoader.html
    
    import gzip
    import numpy as np
    import os
    import struct

    from urllib.request import urlretrieve 

    def load_data(src, num_samples):
        print("Downloading " + src)
        gzfname, h = urlretrieve(src, "./delete.me")
        print("Done.")
        try:
            with gzip.open(gzfname) as gz:
                n = struct.unpack("I", gz.read(4))
                # Read magic number.
                if n[0] != 0x3080000:
                    raise Exception("Invalid file: unexpected magic number.")
                # Read number of entries.
                n = struct.unpack(">I", gz.read(4))[0]
                if n != num_samples:
                    raise Exception(
                        "Invalid file: expected {0} entries.".format(num_samples)
                    )
                crow = struct.unpack(">I", gz.read(4))[0]
                ccol = struct.unpack(">I", gz.read(4))[0]
                if crow != 28 or ccol != 28:
                    raise Exception(
                        "Invalid file: expected 28 rows/cols per image."
                    )
                # Read data.
                res = np.frombuffer(
                    gz.read(num_samples * crow * ccol), dtype=np.uint8
                )
        finally:
            os.remove(gzfname)
        return res.reshape((num_samples, crow, ccol)) / 256


    def load_labels(src, num_samples):
        print("Downloading " + src)
        gzfname, h = urlretrieve(src, "./delete.me")
        print("Done.")
        try:
            with gzip.open(gzfname) as gz:
                n = struct.unpack("I", gz.read(4))
                # Read magic number.
                if n[0] != 0x1080000:
                    raise Exception("Invalid file: unexpected magic number.")
                # Read number of entries.
                n = struct.unpack(">I", gz.read(4))
                if n[0] != num_samples:
                    raise Exception(
                        "Invalid file: expected {0} rows.".format(num_samples)
                    )
                # Read labels.
                res = np.frombuffer(gz.read(num_samples), dtype=np.uint8)
        finally:
            os.remove(gzfname)
        return res.reshape((num_samples))


    def try_download(data_source, label_source, num_samples):
        data = load_data(data_source, num_samples)
        labels = load_labels(label_source, num_samples)
        return data, labels
    
    # Not sure why, but yann lecun's website does no longer support 
    # simple downloader. (e.g. urlretrieve and wget fail, while curl work)
    # Since not everyone has linux, use a mirror from uni server.
    #     server = 'http://yann.lecun.com/exdb/mnist'
    server = 'https://raw.githubusercontent.com/fgnt/mnist/master'
    
    # URLs for the train image and label data
    url_train_image = f'{server}/train-images-idx3-ubyte.gz'
    url_train_labels = f'{server}/train-labels-idx1-ubyte.gz'
    num_train_samples = 60000

    print("Downloading train data")
    train_features, train_labels = try_download(url_train_image, url_train_labels, num_train_samples)

    # URLs for the test image and label data
    url_test_image = f'{server}/t10k-images-idx3-ubyte.gz'
    url_test_labels = f'{server}/t10k-labels-idx1-ubyte.gz'
    num_test_samples = 10000

    print("Downloading test data")
    test_features, test_labels = try_download(url_test_image, url_test_labels, num_test_samples)
    
    return train_features, train_labels, test_features, test_labels


# Call the function to get the data
# Note: I've corrected some bugs in the original file parsing logic (magic numbers)
# to ensure it runs correctly. The function below is the corrected version.

def get_mnist_corrected():
    import gzip, os, struct
    from urllib.request import urlretrieve
    
    def parse_data(src_url, num_samples, is_labels=False):
        print(f"Downloading and parsing {src_url}")
        gzfname, _ = urlretrieve(src_url, "./delete.me")
        try:
            with gzip.open(gzfname, 'rb') as gz:
                magic_number = struct.unpack('>I', gz.read(4))[0]
                expected_magic = 2049 if is_labels else 2051
                if magic_number != expected_magic:
                    raise ValueError(f"Invalid magic number {magic_number}, expected {expected_magic}")
                
                num_items = struct.unpack('>I', gz.read(4))[0]
                if num_items != num_samples:
                    raise ValueError(f"Expected {num_samples} items, file contains {num_items}")

                if is_labels:
                    data = np.frombuffer(gz.read(), dtype=np.uint8)
                else:
                    num_rows = struct.unpack('>I', gz.read(4))[0]
                    num_cols = struct.unpack('>I', gz.read(4))[0]
                    if num_rows != 28 or num_cols != 28:
                        raise ValueError("Images are not 28x28")
                    data = np.frombuffer(gz.read(), dtype=np.uint8).reshape(num_samples, 28, 28)
        finally:
            os.remove(gzfname)
        return data

    server = 'https://raw.githubusercontent.com/fgnt/mnist/master'
    train_count, test_count = 60000, 10000
    
    x_train = parse_data(f'{server}/train-images-idx3-ubyte.gz', train_count) / 255.0
    y_train = parse_data(f'{server}/train-labels-idx1-ubyte.gz', train_count, is_labels=True)
    x_test = parse_data(f'{server}/t10k-images-idx3-ubyte.gz', test_count) / 255.0
    y_test = parse_data(f'{server}/t10k-labels-idx1-ubyte.gz', test_count, is_labels=True)
    
    return x_train, y_train, x_test, y_test
