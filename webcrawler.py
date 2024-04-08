from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from time import sleep

# Replace '/path/to/your/chromedriver' with the actual path to your ChromeDriver
chrome_driver_path = "chromedriver-mac-arm64/chromedriver"
service = Service(executable_path=chrome_driver_path)
driver = webdriver.Chrome(service=service)

def fetch_sub_urls(url):
    driver.get(url)
    sleep(10)  # Respect the crawl-delay as per robots.txt
    elements = driver.find_elements(By.TAG_NAME, "a")
    links = {element.get_attribute('href') for element in elements if 'ece.duke.edu' in element.get_attribute('href')}
    
    # Filtering out any links that lead to disallowed paths or are None
    disallowed = ["/admin/", "/user/", "/search/"]  # Add more based on robots.txt
    filtered_links = {link for link in links if link and not any(dis in link for dis in disallowed)}
    return filtered_links
