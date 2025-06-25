import requests

url = "http://127.0.0.1:5000/analyze"
sample_text = """
Whoopi Goldberg  just claimed being black in America is "no different" than living in Iran.

Seriously?
In Iran, you can be jailed, tortured, or executed for speaking out. Women are beaten for showing hair. 

Protesters are gunned down in the street. There is little to no free speech or civil rights.

In America, you can protest, vote, sue the government, start a business, run for office, and get paid millions to say stupid things on national TV.

Enough with the victim olympics.
"""

response = requests.post(url, json={"text": sample_text})

print("Status Code:", response.status_code)
print("Response JSON:")
print(response.json())
