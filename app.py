import pandas as pd
from fastapi import FastAPI

from lib.constants import MODEL_PATH
from lib.db import *
from lib.utils import *
from lib.transaction import Transaction

conn = create_db_connection()
app = FastAPI()
type_encoder, model, scaler = load_model(MODEL_PATH)

@app.get("/")
async def root():
    return {"message": "Hello World! :)"}

@app.post("/is-fraud")
async def is_fraud(transaction: Transaction):
    insert_transaction(conn, dict(transaction))

    filters = {"step": transaction.step, "nameOrig": transaction.nameOrig }
    prev_transactions = query_transactions(conn, filters)
    if len(list(prev_transactions)) >= 5: return { "isFraud": True }

    transaction = transform(transaction, type_encoder)
    features = to_dataframe(transaction)
    features = normalize(features, scaler)
    prediction = model.predict(features)[0]

    return { "isFraud": bool(prediction) }
