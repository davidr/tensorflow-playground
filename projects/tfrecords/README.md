TFRecord
--------

Given the lack of a fast BigTable reader, and the (understandable) lack of
querying in BigTableReader, it seems that Cloud ML's preferred data ingestion
will be from .tfrecords files on GCS.

Playing around with various ways of writing and reading tfrecords files
