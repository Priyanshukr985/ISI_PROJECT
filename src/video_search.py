import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json


class YouTubeVideoSearch:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("YOUTUBE")

    def is_configured(self):
        return bool(self.api_key)

    def search(self, query, max_results=4, query_prefix="NPTEL", channel_filter=None, fallback_channel_filter=True):
        if not self.api_key:
            return []

        params = {
            "part": "snippet",
            "q": f"{query_prefix} {query}".strip(),
            "type": "video",
            "maxResults": max_results * 3,
            "key": self.api_key,
            "videoEmbeddable": "true",
            "safeSearch": "moderate",
            "relevanceLanguage": "en",
        }
        url = f"https://www.googleapis.com/youtube/v3/search?{urlencode(params)}"
        request = Request(url, headers={"Accept": "application/json"})

        try:
            with urlopen(request, timeout=12) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return []

        videos = []
        for item in payload.get("items", []):
            video_id = item.get("id", {}).get("videoId")
            snippet = item.get("snippet", {})
            channel = snippet.get("channelTitle", "")
            title = snippet.get("title", "")
            description = snippet.get("description", "")
            if not video_id:
                continue
            channel_lower = channel.lower()
            if channel_filter:
                haystack = " ".join([channel_lower, title.lower(), description.lower()])
                if channel_filter.lower() not in haystack:
                    continue
            elif fallback_channel_filter and "nptel" not in channel_lower:
                continue

            thumbnails = snippet.get("thumbnails", {})
            thumb = (
                thumbnails.get("high", {}).get("url")
                or thumbnails.get("medium", {}).get("url")
                or thumbnails.get("default", {}).get("url")
            )

            videos.append(
                {
                    "title": snippet.get("title", "YouTube Video"),
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "embed_url": f"https://www.youtube.com/embed/{video_id}",
                    "channel": channel,
                    "thumbnail": thumb,
                    "published_at": snippet.get("publishedAt", ""),
                    "description": snippet.get("description", ""),
                }
            )

            if len(videos) >= max_results:
                break

        return videos
