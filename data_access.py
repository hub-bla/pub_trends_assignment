import requests
import re
from bs4 import BeautifulSoup

def get_overall_design(accession_id):
    if "GSE" not in accession_id:
        return ""

    url = f'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession_id}'

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        return ""
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    for row in soup.find_all('tr'):
        cells = row.find_all('td')
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True)
            if label == "Overall design":
                return cells[1].get_text(strip=True)
    
    return ""


def fetch_GEO_ids(pmids: list[str]) -> list[str]:
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
    params = {
    "dbfrom": "pubmed",
    "db": "gds",
    "linkname": "pubmed_gds",
    "id": ",".join(pmids),
    "retmode": "json"
    }
    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        response_json = response.json()        
        return response_json["linksets"][0]["linksetdbs"][0]["links"]

    else:
        print(f"Request failed with status code {response.status_code}")
        exit(1)


def fetch_GEO_GSE_GD_ids(GEO_ids: list[str]):
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
    "db": "gds",
    "linkname": "pubmed_gds",
    "id": ",".join(GEO_ids),
    "retmode": "json"
    }
    response = requests.get(BASE_URL, params=params)
    regex = r"Accession:\s*([A-Za-z]+\d+)"
    ids = re.findall(regex, response.text)
    print(len(ids), len(GEO_ids))

    return ids


def get_summaries(GEO_ids):
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {
        "db": "gds",
        "linkname": "pubmed_gds",
        "id": ",".join(GEO_ids),
        "retmode": "json"
    }
    
    response = requests.get(BASE_URL, params=params)
        
    data = response.json()
    rows = []
        
    for geo_id in GEO_ids:
        if geo_id in data.get('result', {}):
            rows.append({
                "geo_id": geo_id,
                "title": data['result'][geo_id].get('title', ""),
                "accession": data['result'][geo_id].get('accession', ""),
                "organism": data['result'][geo_id].get('taxon', ""),
                "summary": data['result'][geo_id].get('summary', ""),
                "pubmedids": data['result'][geo_id].get('pubmedids', []),

            })

    return rows
