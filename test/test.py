import requests

# Fraud
data = {
    "step": 700,
    "type": "TRANSFER",
    "amount": 162326.52,
    "nameOrig": "C1557504343",
    "oldbalanceOrig": 162326.52,
    "newbalanceOrig": 0.00,
    "nameDest": "C404511346",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0
}

# Not fraud
# data = {
#     "step":700,
#     "type":"PAYMENT",
#     "amount":9839.64,
#     "nameOrig":"C1231006815",
#     "oldbalanceOrig":170136.0,
#     "newbalanceOrig":160296.36,
#     "nameDest":"M1979787155",
#     "oldbalanceDest":0.0,
#     "newbalanceDest":0.0
# }

# URL = "http://localhost:8080/is-fraud"
URL = "http://ec2-3-231-160-226.compute-1.amazonaws.com/is-fraud"
response = requests.post(URL, json=data)
print(response.json())
