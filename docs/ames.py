#
#
import os
import sys
import asyncio
from functools import partial
from selenium import webdriver
from selenium.common.exceptions import InvalidSelectorException

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

        target_list = []

        for item in items:
            if not item.get_attribute('href'):
                continue
            self.total_item += 1
            print("++ [item-{}] {}: {}".format(self.total_item, item.text, item.get_attribute('href')))
            target_list.append(item.get_attribute('href'))

        future = asyncio.gather(*[partial(self.foreach_item, a) for a in target_list], loop=self.loop)


    @asyncio.coroutine
    def foreach_item(self, url):
        yield from [self.drv.get(url)]
        index = ['Summary', 'Technology Details', 'Benefits',\
                'Commercial Applications', 'Patents', 'For More Information']

        items = self.drv.find_elements_by_xpath('//p')

        if not items:
            return

        target_list = []

        for item in items:
            if not item.text:
                continue
            target_list.append(item.text)

        print('++ [info] {} {}'.format(len(target_list), len((index))))
            
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


if __name__ == '__main__':
    Ames.start()
