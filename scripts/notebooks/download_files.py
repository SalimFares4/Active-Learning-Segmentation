import gdown

import urllib.request
main_path = "/root/Master_Thesis/"

example_path = main_path + "data/tumor.png"

# url = "https://drive.google.com/file/d/1hBiUe6bAY7kKQ9bjO9r19ZPjFn2bxeAy/view?usp=sharing"
url = "https://drive.google.com/file/d/1hBiUe6bAY7kKQ9bjO9r19ZPjFn2bxeAy/view?usp=sharing"
gdown.download(url, example_path, fuzzy=True, quiet=False)

# urllib.request.urlretrieve(url, filename=example_path)