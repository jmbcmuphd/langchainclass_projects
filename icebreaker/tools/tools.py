from langchain_community.tools.tavily_search import TavilySearchResults


def get_profile_url_tavily(name: str):
    """Searched for LinkedIn or Twitter Profile page."""
    search = TavilySearchResults()
    res = search.run(f"{name}")
    return res[0]["url"]
