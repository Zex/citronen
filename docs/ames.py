#
#
import os
import sys
import asyncio
import ujson
from copy import deepcopy
import re
from functools import partial
from selenium import webdriver
from selenium.common.exceptions import InvalidSelectorException
from src.etl.unifier.common import iter_line

BASE_URL = "https://www.nasa.gov/ames-partnerships/patent-portfolio"


class Ames(object):

    def __init__(self):
        super(Ames, self).__init__()
        self.drv = webdriver.PhantomJS()
        self.total_items = 0
        self.loop = asyncio.get_event_loop()
        print(dir(self.drv))

    @asyncio.coroutine
    def foreach_list(self, url):
        yield from [self.drv.get(url)]
        items = self.drv.find_elements_by_xpath('//p/a')

        if not items:
            return

        target_list = set()
        for item in items:
            if not item.get_attribute('href'):
                continue
            a = item.get_attribute('href')
            self.total_items += 1
            print("++ [item-{}] {}: {}".format(self.total_items, item.text, a))
            target_list.add(a)

        [asyncio.ensure_future(self.foreach_item(a), loop=self.loop) for a in target_list]

    @asyncio.coroutine
    def foreach_item(self, url):
        yield from [self.drv.get(url)]

        items = self.drv.find_elements_by_xpath('//p')

        if not items:
            return
        
        meta = {
                'source_url': url,
                'source': "NASA AMES",
                }

        for i, item in enumerate(items):
            if not item.text:
                continue
            print('++ [info] {} {}'.format(url, item.text))
            meta.update({'p_{}'.format(i): item.text.strip()})

        items = self.drv.find_elements_by_xpath("//h3[@class='title']")

        if items:
            item = items[0]
            meta.update({'title': item.text})

        with open('meta.json', 'a') as fd:
            fd.write(ujson.dumps(meta)+'\n')
            
    @classmethod
    def start(cls):
        obj = cls()

        obj.drv.get(BASE_URL)
        items = obj.drv.find_elements_by_xpath('//p/a')
        target_list = []

        if not items:
            return

        for item in items:
            if not item.get_attribute('href'):
                continue
            print("++ [add] {}: {}".format(item.text, item.get_attribute('href')))
            target_list.append(item.get_attribute('href'))

        future = asyncio.gather(*[obj.foreach_list(a) for a in target_list])
        obj.loop.run_until_complete(future)

def parse_paragraph():
    path = 'ames.json'
    index = ['Summary', 'Technology Details', 'Benefits',\
                'Commercial Applications', 'Patents', 'For More Information']

    for line in iter_line(path):
        meta = ujson.loads(line)
        updated = deepcopy(meta)                

        for k, v in meta.items():
            done = False
            for ind in index:
                grp = v.match(ind)
                if grp:
                    updated.update({ind: v[grp.span()[1]+1:].strip()})
                    done = True
                    break

            if done:
                continue

            grp = meta.get('p_{}'.format(int(k.split('_')[1])-1), '')

            if meta.get(grp, '').strip() in index:
                updated.update({meta.get(grp).strip(): v})

        [print(k, '=>', v) for k, v in meta.items()]


if __name__ == '__main__':
    Ames.start()
