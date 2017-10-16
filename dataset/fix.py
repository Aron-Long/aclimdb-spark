import sklearn.datasets as ds

x = ds.load_svmlight_file("aclImdb/train/labeledBow.feat")
ds.dump_svmlight_file(x[0], x[1], "aclImdb/train/labeledBow-1-index.feat", zero_based=False)

x = ds.load_svmlight_file("aclImdb/test/labeledBow.feat")
ds.dump_svmlight_file(x[0], x[1], "aclImdb/test/labeledBow-1-index.feat", zero_based=False)


x = ds.load_svmlight_file("aclImdb/train/unsupBow.feat")
ds.dump_svmlight_file(x[0], x[1], "aclImdb/train/unsupBow-1-index.feat", zero_based=False)
