from pymongo import MongoClient

import settings

mongo_host = settings.MONGO_HOST
mongo_port = settings.MONGO_PORT
mongo_user = settings.MONGO_USER
mongo_pswd = settings.MONGO_PSWD

uri = f"mongodb://{mongo_user}:{mongo_pswd}@{mongo_host}:{mongo_port}/?unicode_decode_error_handler=ignore"
client = MongoClient(uri)

db_names = {}
for i, db in enumerate(client.list_databases()):
    db_names[i] = db['name']


def get_collections(db_no):
    db = client.get_database(db_names[db_no])
    coll_names = {}
    for i, coll in enumerate(db.list_collection_names()):
        coll_names[i] = coll
    return coll_names
