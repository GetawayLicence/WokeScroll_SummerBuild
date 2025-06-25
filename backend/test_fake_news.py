import requests

sample = {
    "title": "Trump administration issues new rules on U.S. visa waivers"
}

response = requests.post("http://127.0.0.1:5000/predict", json=sample)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
