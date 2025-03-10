import requests
import json
import sys

def test_api(url):
    """Test the API by making a GET request to the specified URL."""
    print(f"Testing API at: {url}")
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"Response (JSON): {json.dumps(data, indent=2)}")
            except json.JSONDecodeError:
                print(f"Response (Text): {response.text[:500]}")
        else:
            print(f"Response: {response.text[:500]}")
            
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to {url}")
    except Exception as e:
        print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    # Test both path variations
    test_api("http://localhost:5000/available_models")
    print("\n" + "-"*50 + "\n")
    test_api("http://localhost:5000/api/available_models")
