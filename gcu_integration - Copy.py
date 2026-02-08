from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
import time
import re

class GCUPortal:
    def __init__(self):
        print("Initializing GCU Portal connection...")
        self.service = ChromeService(ChromeDriverManager().install())
        self.options = webdriver.ChromeOptions()
        self.options.add_argument("--headless=new") 
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36")
        self.driver = None

    def login(self, username, password):
        # Check if already logged in and driver is active
        if self.driver:
            try:
                # fast check: if we are on dashboard, we are good
                if "dashboard" in self.driver.current_url.lower() or "home" in self.driver.current_url.lower():
                    print("Session active. Skipping login.")
                    return True
            except:
                print("Driver unresponsive, restarting...")
                self.driver.quit()
                self.driver = None

        try:
            if not self.driver:
                self.driver = webdriver.Chrome(service=self.service, options=self.options)
            
            url = "https://grms.gcu.edu.in/auth/sign-in/"
            print(f"Navigating to {url}...")
            self.driver.get(url)

            wait = WebDriverWait(self.driver, 20)
            
            # Mimic human behavior: Click then type
            print("Entering username...")
            username_field = wait.until(EC.element_to_be_clickable((By.NAME, "username")))
            username_field.click()
            username_field.send_keys(username)
            time.sleep(0.5)

            print("Entering password...")
            password_field = self.driver.find_element(By.NAME, "password")
            password_field.click()
            password_field.send_keys(password)
            time.sleep(1) # Wait for React state to update
            
            # Find and click submit button
            print("Clicking submit...")
            submit_btn = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
            
            # Check if disabled
            if not submit_btn.is_enabled():
                print("Button is disabled, waiting...")
                time.sleep(1)
            
            if submit_btn.is_enabled():
                submit_btn.click()
            else:
                 # Force click with JS if still disabled (fallback)
                print("Button still disabled. Attempting JS click...")
                self.driver.execute_script("arguments[0].click();", submit_btn)

            # Wait for dashboard
            print("Waiting for redirect...")
            time.sleep(8)
            
            curr_url = self.driver.current_url.lower()
            print(f"Current URL after login attempt: {curr_url}")

            if "dashboard" in curr_url or "home" in curr_url:
                print("Login successful.")
                return True
            else:
                print("Login failed or timed out.")
                return False
        except Exception as e:
            print(f"Login error: {e}")
            # If login fails, kill driver so next try starts fresh
            if self.driver:
                try:
                    self.driver.quit()
                except:
                    pass
                self.driver = None
            return False

    def get_student_data(self):
        """
        Scrapes the dashboard for likely student data.
        Returns a dictionary or formatted string.
        """
        if not self.driver:
            return "Error: Not logged in."

        try:
            # Ensure we are on the dashboard
            if "dashboard" not in self.driver.current_url.lower():
                print("Not on dashboard, navigating...")
                self.driver.get("https://grms.gcu.edu.in/dashboard/")
                time.sleep(3)

            print("Scraping dashboard data...")
            # Get full text content
            body_text = self.driver.find_element(By.TAG_NAME, "body").text
            
            # DEBUG: Save to file to see what we actually got
            print(f"Current URL: {self.driver.current_url}")
            with open("dashboard_dump.txt", "w", encoding="utf-8") as f:
                f.write(f"URL: {self.driver.current_url}\n\n")
                f.write(body_text)
            
            # Simple keyword extraction (Customise based on actual dashboard text)
            data = {}
            
            # Look for "Overall Attendance Status" followed by a number
            # Pattern: "Overall Attendance Status\n0"
            attendance_match = re.search(r"Overall Attendance Status\s*\n\s*(\d+%?)", body_text, re.IGNORECASE)
            if attendance_match:
                data['Attendance'] = attendance_match.group(1) + "%" # Assuming it is a percentage if just a number
            else:
                 # Fallback: check for "0 Present"
                 present_match = re.search(r"(\d+)\s*Present", body_text)
                 if present_match:
                     data['Attendance'] = f"{present_match.group(1)} Days Present"

            # Look for Fee
            # Pattern: "Pending Fee\n₹0"
            fee_match = re.search(r"Pending Fee\s*\n\s*([₹\d,]+)", body_text, re.IGNORECASE)
            if fee_match:
                data['Pending Fee'] = fee_match.group(1)
            
            # Look for Assignments
            # Just count them or list first few
            assignments = re.findall(r"(.*?)\nDue:", body_text)
            if assignments:
                data['Assignments Due'] = ", ".join(assignments[:3])

            # Look for Recent Events / Upcoming Events
            # Pattern: "Recent Events... (Events listing) ... "
            # We'll try to capture lines after "Recent Events" until we hit another section key or end of short list
            try:
                if "Recent Events" in body_text:
                    # simplistic extraction: get next few lines that look like event titles (not dates or status)
                    # This is tricky with just text, but let's try to grab a chunk
                    events_section = re.search(r"Recent Events(.*?)(?:Overall Attendance|Payment|Assignments|$)", body_text, re.DOTALL | re.IGNORECASE)
                    if events_section:
                        raw_events = events_section.group(1).strip().split('\n')
                        # Filter out common UI words and date lines (which seem to start with 'conference jan/feb...')
                        clean_events = []
                        for line in raw_events:
                            line = line.strip()
                            # Skip short lines, "View All", "Ongoing", "Scheduled", "Completed" headers
                            if len(line) < 3 or "View All" in line:
                                continue
                            if line in ["Ongoing", "Scheduled", "Completed", "Recent Events"]:
                                continue
                            # Skip the date lines which seem to start with lowercase 'conference' followed by months
                            # Heuristic: if it starts with 'conference' and has a digit, it's likely the date line
                            if line.lower().startswith("conference") and any(c.isdigit() for c in line):
                                continue
                                
                            clean_events.append(line)

                        if clean_events:
                            data['Upcoming Events'] = ", ".join(clean_events[:5])
            except Exception as e:
                print(f"Event parsing error: {e}")

            result_str = "🎓 **Student Dashboard Data**\n"
            if not data:
                result_str += f"Dashboard accessed, but no specific data found (e.g. Attendance is 0 or hidden).\nRaw text snippet:\n{body_text[:200]}..."
            else:
                for key, value in data.items():
                    result_str += f"- **{key}**: {value}\n"
            
            return result_str

        except Exception as e:
            return f"Error extracting data: {str(e)}"

    def get_events(self):
        """
        Navigate to the Events page and scrape upcoming events.
        Returns formatted string of events.
        """
        if not self.driver:
            return "Error: Not logged in."

        try:
            print("Navigating to Events page...")
            # Navigate to events list - use the sidebar menu Events section
            self.driver.get("https://grms.gcu.edu.in/events/")
            time.sleep(3)
            
            # Try to click on "Upcoming Events" tab if available
            try:
                upcoming_tab = self.driver.find_element(By.XPATH, "//button[contains(text(),'Upcoming Events')] | //a[contains(text(),'Upcoming Events')]")
                upcoming_tab.click()
                time.sleep(2)
            except:
                print("Could not find Upcoming Events tab, using default view")
            
            body_text = self.driver.find_element(By.TAG_NAME, "body").text
            
            # DEBUG
            print(f"Events page URL: {self.driver.current_url}")
            with open("events_dump.txt", "w", encoding="utf-8") as f:
                f.write(f"URL: {self.driver.current_url}\n\n")
                f.write(body_text)
            
            # Extract events - parse the list view
            # Based on user's screenshot, the structure seems to be:
            # Date (like "11 Feb 2026") followed by Event Name, then location, then status
            events = []
            lines = body_text.split('\n')
            
            # Look for event patterns - event names are usually descriptive titles
            skip_words = ["All Events", "Upcoming Events", "Running Events", "Completed Events", 
                          "Search events...", "Search", "Events", "Events List", "Register Now",
                          "days to register", "Auditorium", "OMR Campus", "Sponsored", 
                          "www.", "Event Type", "Date From", "Till Date", "Event Location",
                          "Registration Details", "Close Date", "Event Agenda", "Speakers",
                          "No Agenda", "No Speakers", "Organizer", "WORKSHOP", "12:00 AM"]
            
            for line in lines:
                line = line.strip()
                if not line or len(line) < 5:
                    continue
                
                # Skip UI/metadata lines
                if any(skip in line for skip in skip_words):
                    continue
                
                # Skip date-only lines (like "11 Feb 2026")
                if re.match(r'^\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}$', line, re.IGNORECASE):
                    continue
                    
                # Skip short date formats
                if re.match(r'^\d{1,2}\s+(Feb|Jan|Mar)', line):
                    continue
                
                # Skip HTML tags
                if line.startswith('<') and line.endswith('>'):
                    continue
                if '<p>' in line or '</p>' in line:
                    continue
                    
                # Skip numbers only
                if line.isdigit():
                    continue
                
                # Event names are usually longer and descriptive
                if len(line) > 10 and not line.startswith('₹') and not line.startswith('0'):
                    events.append(line)
            
            if events:
                result = "📅 **Upcoming Events from GRMS**\n"
                # Show up to 5 unique events
                seen = set()
                count = 0
                for event in events:
                    if event not in seen and count < 5:
                        result += f"- {event}\n"
                        seen.add(event)
                        count += 1
                return result
            else:
                return f"No upcoming events found.\n(Debug: Raw dump saved to events_dump.txt)"
                
        except Exception as e:
            return f"Error getting events: {str(e)}"

    def get_attendance(self):
        """
        Navigate to the Attendance page and scrape attendance data.
        """
        if not self.driver:
            return "Error: Not logged in."

        try:
            print("Navigating to Attendance page...")
            self.driver.get("https://grms.gcu.edu.in/attendance/")
            time.sleep(4)
            
            body_text = self.driver.find_element(By.TAG_NAME, "body").text
            
            # DEBUG
            print(f"Attendance page URL: {self.driver.current_url}")
            with open("attendance_dump.txt", "w", encoding="utf-8") as f:
                f.write(f"URL: {self.driver.current_url}\n\n")
                f.write(body_text)
            
            # Look for attendance percentage or stats
            result = "📊 **Attendance from GRMS**\n"
            
            # Try to find percentage
            pct_match = re.search(r"(\d+(\.\d+)?)\s*%", body_text)
            if pct_match:
                result += f"- Overall Attendance: {pct_match.group(0)}\n"
            
            # Try to find Present/Absent counts
            present_match = re.search(r"(\d+)\s*Present", body_text, re.IGNORECASE)
            absent_match = re.search(r"(\d+)\s*Absent", body_text, re.IGNORECASE)
            if present_match:
                result += f"- Days Present: {present_match.group(1)}\n"
            if absent_match:
                result += f"- Days Absent: {absent_match.group(1)}\n"
                
            if "No Data" in body_text or ("0" in result and "Present" not in body_text):
                result += "(No attendance data recorded yet)\n"
            
            return result
                
        except Exception as e:
            return f"Error getting attendance: {str(e)}"

    def close(self):
        if self.driver:
            print("Closing GCU Portal driver...")
            self.driver.quit()
            self.driver = None

