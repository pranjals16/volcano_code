from bs4 import BeautifulSoup
import mechanize
import random
import time
import pandas as pd


PRE_LINK = "https://www.theatlantic.com"

br = mechanize.Browser()
br.set_handle_robots(False)
br.set_handle_equiv(False)
br.addheaders = [('user-agent', '   Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.2.3) Gecko/20100423 Ubuntu/10.04 (lucid) Firefox/3.6.3'),
('accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8')]

res = []
for counter in range(1, 692):
	link = "https://www.theatlantic.com/technology/?page=" + str(counter)
	br.open(link)
	soup = BeautifulSoup(br.response().read(), "lxml")
	print counter
	for href in soup.find_all('h2'):
		if href.parent.name == 'a':
			NEW_LINK = PRE_LINK + href.parent.get('href')
			if "archive" in NEW_LINK:
				res.append(NEW_LINK)
	counter += 1


with open("atlantic_technology_links.txt", "w") as f:
	for item in res:
		f.write(item + "\n")

res = []
with open("atlantic_technology_links.txt", "rb") as f:
	for line in f:
		res.append(line.strip())

for i in range(10248, len(res)):
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
		full_text += soup.find("h1", {"itemprop":"headline"}).text + "\n" + soup.find("p", {"itemprop":"description"}).text + "\n"
	except:
		full_text += soup.find("title").text.replace(" - The Atlantic", "") + "\n"
	try:
		sections = soup.findAll("section", {"itemprop":"articleBody"})
	except:
		sections = soup.findAll("div", {"itemprop":"articleBody"})
	if len(sections) == 0:
		sections = soup.findAll("div", {"class":"article-body"})
	for item in sections:
		full_text += item.text
		full_text += "\n"
	with open("data/atlantic/technology/" + str(i + 1) + ".txt", "w") as f:
		f.write(link + "\n")
		f.write(full_text.encode("utf-8"))
		f.write("\n")
	time.sleep(random.randint(1, 2))
