import requests
import pandas as pd
import xml.etree.ElementTree as ET
import time

def fetch_arxiv_data(query="machine learning", total_results=500, batch_size=500):
    papers = []

    for start in range(0, total_results, batch_size):
        print(f"Fetching papers {start} to {start + batch_size}...")

        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start={start}&max_results={batch_size}"
        
        response = requests.get(url)

        if response.status_code != 200:
            print("❌ Error fetching data")
            continue

        root = ET.fromstring(response.content)

        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text
            summary = entry.find("{http://www.w3.org/2005/Atom}summary").text
            link = entry.find("{http://www.w3.org/2005/Atom}id").text

            papers.append({
                "title": title.strip(),
                "abstract": summary.strip(),
                "link": link
            })

        time.sleep(3)  # VERY IMPORTANT (avoid rate limits)

    df = pd.DataFrame(papers)
    df.to_csv("papers.csv", index=False)

    print(f"\n✅ Total papers collected: {len(df)}")

if __name__ == "__main__":
    fetch_arxiv_data()