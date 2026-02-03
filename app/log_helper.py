import yaml
import random
import requests
from threading import Thread

config_file_path = "app/config.yaml"

with open(config_file_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# matomo site id, i.e. the site for which this event will be logged on matomo
matomo_site_id = config["matomo_site_id"]
matomo_url = config["matomo_url"]
matomo_ssl_verify = config["matomo_ssl_verify"]


def matomo_track_event(
    user: str = "na",
    category: str = "na",
    action: str = "na",
    name: str = "na",
    value: int = 0,
    threaded: bool = True,
) -> None:
    """
    log a user event, integrating matomo event tracking w/streamlit
    use this function in conjunction with if statements on the streamlit app to log user activity
    :param user: (str) matomo param "user", i.e. the name/contact of the user that triggered this event (default: blank)
    :param category: (str) matomo param "category", i.e. the category of this event (default: blank)
    :param action: (str) matomo param "action", i.e. the action associated with this event (default: blank)
    :param name: (str) matomo param "name", i.e. the name of the element associated with this event (default: blank)
    :param value: (int) matomo param "value", i.e. if any, the value generated as a result of this event (default: blank); using floats not recommended - matomo does accept floats but if the float is too long, the value is automatically zeroed out (kinda silly)
    :return : (Future[Response]) the status code of the HTTP request (200 is the desired status code)
    """

    if threaded:
        thread = Thread(
            target=matomo_track_event, args=(user, category, action, name, value, False)
        )
        thread.start()
        return

    # required for matomo tracking to work...
    tracking_enabled = 1
    avoid_caching = random.randint(0, 999)

    data = {
        "rec": tracking_enabled,
        "rand": avoid_caching,
        "idsite": matomo_site_id,
        "uid": user,
        "e_c": category,
        "e_a": action,
        "e_n": name,
        "e_v": value,
    }

    try:
        requests.post(url=matomo_url, data=data, verify=matomo_ssl_verify)
    except Exception as e:
        print(e)
        print(
            "matomo url: %s, site id:%d, uid: %s, cat: %s, act: %s, name: %s, value: %s."
            % (matomo_url, matomo_site_id, user, category, action, name, value)
        )
