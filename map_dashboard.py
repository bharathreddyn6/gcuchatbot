from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
import time

# Credentials
USERNAME = "22btcs128@gcu.edu.in"
PASSWORD = "Soz38610"
URL = "https://grms.gcu.edu.in/auth/sign-in/"

try:
    print("Setting up Chrome Driver...")
    service = ChromeService(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new") 
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    
    driver = webdriver.Chrome(service=service, options=options)
    
    print(f"Navigating to {URL}...")
    driver.get(URL)
    
    wait = WebDriverWait(driver, 20)
    username_field = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[name='email'], input[name='username'], input[type='email']")))
    username_field.send_keys(USERNAME)
    
    password_field = driver.find_element(By.CSS_SELECTOR, "input[name='password'], input[type='password']")
    password_field.send_keys(PASSWORD)
    password_field.send_keys(Keys.RETURN)
    
    time.sleep(8) # Wait for full load
    
    print(f"Current URL: {driver.current_url}")
    
    # Get all links
    links = driver.find_elements(By.TAG_NAME, "a")
    print(f"Found {len(links)} links on dashboard:")
    for link in links:
        try:
            href = link.get_attribute('href')
            text = link.text.strip().replace("\n", " ")
            if href and len(text) > 2:
                print(f"Text: {text} | HREF: {href}")
        except:
            pass
            
    # Also look for cards or buttons that might be 'Results' or 'Attendance'
    print("\nLooking for keywords...")
    keywords = ["Attendance", "Result", "Marks", "Grade", "Exam", "Fee"]
    for kw in keywords:
        elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{kw}')]")
        if elements:
            print(f"Found '{kw}': {len(elements)} times")
            for el in elements[:3]:
                print(f"  - Tag: {el.tag_name}, Class: {el.get_attribute('class')}")
    
    driver.quit()

except Exception as e:
    print(f"An error occurred: {e}")
    if 'driver' in locals():
        driver.quit()
