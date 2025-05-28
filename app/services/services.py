import requests
from serpapi import GoogleSearch
from settings import SERP_API_KEY, WEATHER_API_KEY, TOMTOM_API_KEY


def search_web(query, top_k=5):
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERP_API_KEY,  # Thay bằng API key SerpAPI của bạn
        "hl": "vi",
        "num": top_k
    }

    search = GoogleSearch(params)
    results = search.get_dict().get("organic_results")

    data = []
    for result in results:
        data.append({
            "position": result.get("position"),
            "title": result.get("title"),
            "snippet": result.get("snippet"),
            "date": result.get("date", None),
        })

    return "\n".join(f"({item["position"]}) Tiêu đề thông tin: {item["title"]}. Nội dung thông tin: {item["snippet"]}. Ngày đăng: {item["date"]}" for item in data)


def get_temperature_and_weather():
    response = requests.get(
        "https://api.weatherapi.com/v1/current.json",
        params={
            "key": WEATHER_API_KEY,
            "q": f"Da Nang",
            "lang": "vi"
        }
    )
    data = response.json()
    data = {
        "location": {
            "name": data["location"]["name"],
            "region": data["location"]["region"],
            "country": data["location"]["country"],
            "localtime": data["location"]["localtime"]
        },
        "current": {
            "temp_c": data["current"]["temp_c"],
            "condition": {
                "text": data["current"]["condition"]["text"],
            },
        }
    }
    return data


def get_traffic_data(address):
    api_key = TOMTOM_API_KEY
    url = f"https://api.tomtom.com/search/2/geocode/{address}.json"

    params = {
        "key": api_key
    }

    res = requests.get(url, params=params)
    data = res.json()
    address_name = data["results"][0]["address"]["freeformAddress"]
    radius = 1000  # bán kính 1km
    lat = data["results"][0]["position"]["lat"]
    lon = data["results"][0]["position"]["lon"]

    url = "https://api.tomtom.com/traffic/services/5/incidentDetails"
    params = {
        "point": f"{lat},{lon}",
        "radius": radius,
        "key": api_key,
        "language": "en-US",
        "bbox": f"{lat - 0.01},{lon - 0.01},{lat + 0.01},{lon + 0.01}",
    }

    response = requests.get(url, params=params)
    data = response.json()

    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json"
    params = {
        "point": f"{lat},{lon}",
        "key": api_key
    }

    response = requests.get(url, params=params)
    flow_data = response.json()['flowSegmentData']

    road_type = flow_data['frc']
    current_speed = flow_data["currentSpeed"]
    free_speed = flow_data["freeFlowSpeed"]
    current_time = flow_data["currentTravelTime"]
    free_time = flow_data["freeFlowTravelTime"]
    confidence = flow_data["confidence"]
    roadClosure = flow_data["roadClosure"]
    # Phân tích
    congestion_ratio = current_time / free_time
    if congestion_ratio < 1.2:
        status = "Bình thường"
    elif congestion_ratio < 1.5:
        status = "Hơi đông"
    elif congestion_ratio < 2.0:
        status = "Đang kẹt xe"
    else:
        status = "Kẹt nghiêm trọng"
    match road_type:
        case "FRC1":
            road_type = "Đường cao tốc"
        case "FRC2":
            road_type = "Đường quốc lộ lớn"
        case "FRC3":
            road_type = "Đường thành phố lớn"
        case "FRC4":
            road_type = "Đường cấp địa phương"
        case "FRC5":
            road_type = "Đường dân sinh"
        case "FRC6":
            road_type = "Ngõ nhỏ, hẻm"
        case "FRC7":
            road_type = "lối đi bộ, đường trong khuôn viên"
        case _:
            road_type = "Không xác định"
    res = {
        "Tên vị trí": address_name,
        "Tình trạng của đường": status,
        "Tốc độ xe hiện tại": current_speed,
        "Tốc độ xe khi đường vắng": free_speed,
        "Loại đường": road_type,
    }
    incidents = []

    # In thông tin sự cố
    for incident in data.get("incidents", []):
        incident_info = {
            "description": incident.get("description","Không rõ"),
            "incidentCategory": incident.get("incidentCategory","Không rõ"),
            "severity": incident.get("severity", "Không rõ"),
            "frc": incident.get("frc", "Không rõ"),
        }
        incidents.append(incident_info)
    res["Sự cố giao thông"] = incidents
    return res

