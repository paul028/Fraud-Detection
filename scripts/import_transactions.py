import sqlalchemy
from sqlalchemy import String
import pandas as pd
from datetime import datetime
from lib.constants import DATABASE_URL

engine = sqlalchemy.create_engine(DATABASE_URL)
connection = engine.connect()
transactions = pd.read_csv("data/transactions_train.csv")
dtypes = {
    'step': sqlalchemy.types.INTEGER(),
    'type': sqlalchemy.types.NVARCHAR(length=255),
    'nameOrig': sqlalchemy.types.NVARCHAR(length=255),
    'nameDest': sqlalchemy.types.NVARCHAR(length=255),
    'isFraud': sqlalchemy.types.Boolean
}

print("Importing records to database")
start_time = datetime.now()

transactions.to_sql(
    name='transactions',
    chunksize=10000,
    con=connection,
    if_exists='replace',
    dtype=dtypes,
    method='multi',
    index=False
)

print("Total time: {}".format(datetime.now() - start_time))
print(pd.read_sql("SELECT * from transactions limit 100;", connection))
