import requests

def compile_c_code(code):
    clientId = 'af116396f4cab55164b10ac0aa7bd424'  # JDoodle clientId
    clientSecret = '72e373a4e4d6547eda8054627144f0217a58c4afc1c3c193d73d8f41023eeea6'  # JDoodle clientSecret
    script = code
    language = "c"
    versionIndex = "0"  # Change if you use a different version
    
    data = {
        "clientId": clientId,
        "clientSecret": clientSecret,
        "script": script,
        "language": language,
        "versionIndex": versionIndex
    }
    
    response = requests.post("https://api.jdoodle.com/v1/execute", json=data)
    result = response.json()
    
    # Extracting output from result
    output = result.get("output", "No output")
    statusCode = result.get("statusCode", "No status code")
    
    return f"Status Code: {statusCode}\n{output}"
