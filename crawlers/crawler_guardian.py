from bs4 import BeautifulSoup
import mechanize
import random
import time
import cookielib
import pandas as pd


br = mechanize.Browser()
cj = cookielib.LWPCookieJar()
br.set_cookiejar(cj)

# Browser options
br.set_handle_equiv(True)
br.set_handle_gzip(True)
br.set_handle_redirect(True)
br.set_handle_referer(True)
br.set_handle_robots(False)
br.set_handle_refresh(mechanize._http.HTTPRefreshProcessor(), max_time=1)
br.addheaders = [('user-agent', '   Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.2.3) Gecko/20100423 Ubuntu/10.04 (lucid) Firefox/3.6.3'),
('accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8')]

res = []
for counter in range(1, 920):
	link = "https://www.theguardian.com/uk/culture?page=" + str(counter)
	br.open(link)
	soup = BeautifulSoup(br.response().read(), "lxml")
	print counter
	for href in soup.findAll("a",{"data-link-name":"article"}):
		res.append(href['href'])
	counter += 1


with open("guardian_global-development_links.txt", "w") as f:
	for item in res:
		f.write(item + "\n")

res = []
with open("guardian_global-development_links.txt", "rb") as f:
	for line in f:
		res.append(line.strip())

for i in range(18239, len(res)):
	link = res[i]
	print i + 1
	try:
		br.open(link)
	except:
		time.sleep(5)
		br.open(link)
	soup = BeautifulSoup(br.response().read(), "lxml")
	full_text = ""
	try:
		full_text += soup.find("h1", {"articleprop":"headline"}).get_text(strip = True) + "\n"
	except:
		try:
			full_text += soup.find("h1", {"itemprop":"headline"}).get_text(strip = True) + "\n"
		except:
			print "No Headline"
	try:
		full_text += soup.find("meta", {"itemprop":"description"})['content'] + "\n"
	except:
		try:
			full_text += soup.find("name", {"itemprop":"description"})['content'] + "\n"
		except:
			print "No description"

	sections = soup.findAll("div", {"itemprop":"articleBody"})
	for item in sections:
		full_text += item.get_text(strip = True)
		full_text += "\n"
		with open("data/guardian/global-development/" + str(i + 1) + ".txt", "w") as f:
			f.write(link + "\n")
			f.write(full_text.encode("utf-8"))
			f.write("\n")
		time.sleep(random.randint(1, 2))