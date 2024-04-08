from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from time import sleep
import os
from webcrawler import fetch_sub_urls

# Replace '/path/to/your/chromedriver' with the actual path to your ChromeDriver
chrome_driver_path = "chromedriver-mac-arm64/chromedriver"
service = Service(executable_path=chrome_driver_path)
driver = webdriver.Chrome(service=service)

def scrape_to_text(url_list):
    for url in url_list:
        try:
            driver.get(url)
            sleep(10)  # Respect the website's crawl delay as per robots.txt

            # Example: extract all paragraph texts; adjust as needed
            # Here, we're collecting all text from <p> elements; you might need to modify this based on the site's structure
            content_element = driver.find_element(By.ID, "main-content")
            page_content = content_element.text

            # Generate filename from URL
            filename = url.rstrip('/').split('/')[-1] + ".txt"
            if not filename:
                filename = "index.txt"  # Default name if last URL part is empty
            
            # Save content to a .txt file
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(page_content)
                print(f"Content from {url} saved to {filename}")

        except Exception as e:
            print(f"Error scraping {url}: {e}")

# URL to start with
start_url = "https://ece.duke.edu/"
url_list = fetch_sub_urls(start_url)
# Create a directory for the saved files
os.makedirs("scraped_content", exist_ok=True)
os.chdir("scraped_content")
scrape_to_text(url_list)

driver.quit()