# %%
import requests
from bs4 import BeautifulSoup
import os

all_names = []

# Define the directory where you want to save the zip files
# TODO: change it to output folder
output_directory = "input/bollywood"  # Replace with your target directory

# Ensure the directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

progress_file = os.path.join(output_directory, "progress.txt")

if os.path.exists(progress_file):
    with open(progress_file, 'r') as f:
        start_page = int(f.read().strip())
else:
    start_page = 0

for page in range(54, 386):
    try:
        print("downloading files for curr page {}".format(page))
        if page == 0:
            url = "https://www.bollynook.com/en/bollywood-movie-subtitles/"
        else:
            url = "https://www.bollynook.com/en/bollywood-movie-subtitles/1/{}".format(page+1)

        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'lxml')
        soup = soup.find("div", class_="paddingContent")
        language = soup.find_all('a', class_="lang-tooltip")
        link = soup.find_all('a', class_="btn btn-mini btn-danger")

        names = []
        web_links = []
        years = []
        download_links = []

        for i in range(len(language)):
            if language[i]["data-original-title"] == "English":
                download_link = "https://www.bollynook.com" + link[i]["href"]
                name = download_link.split('/')[-2]
                if name not in all_names and "series" not in name and "quantico" not in name:
                    all_names.append(name)
                    names.append(name)
                    web_links.append(download_link)

        for i in web_links:
            response = requests.get(i)
            soup = BeautifulSoup(response.content, 'lxml')
            download_link = soup.find('a', class_="btn btn-info downloads")['href']
            download_link = "https://www.bollynook.com" + download_link
            download_links.append(download_link)

            info = soup.find(class_="other-info").find_all('li')
            years.append(info[4].text.strip())  # Strip any extra spaces

        for i in range(len(download_links)):
            # Create the filename with the release year
            movie_name_with_year = "{}_{}.zip".format(names[i], years[i])
            output_path = os.path.join(output_directory, movie_name_with_year)

            # Download the file
            url = download_links[i]
            r = requests.get(url, allow_redirects=True)
            open(output_path, 'wb').write(r.content)
            print("Downloaded: {}".format(output_path))
        
        with open(progress_file, 'w') as f:
            f.write(str(page))

    except Exception as e:
        print("An error occurred on page {}: {}".format(page, e))
        with open(progress_file, 'w') as f:
            f.write(str(page))
        break



