import urllib.request as libreq

with libreq.urlopen('http://export.arxiv.org/api/query?search_query=all:artificial%20intelligence') as url:
    r = url.read()
print(r)