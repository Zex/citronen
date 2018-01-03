# Prediction consumer for each data type
# Author: Zex Li <top_zlynch@yahoo.com>
import os
import ujson
import numpy as np
import pandas as pd
from boto3.dynamodb.conditions import Key, Attr
from julian.output.consumer import Prediction as PC
from julian.common.config import get_config
from julian.common.topic import Topic
from src.dynamodb.core.broker import get_brk
from src.dynamodb.tables.journal_article import JournalArticle
from src.dynamodb.tables.tech_classi import TechClassi


class Article(PC):
    """Prediction consumer for article"""
    def __init__(self, **kwargs):
        super(Article, self).__init__(topics=(Topic.PREDICT_TECH,),  **kwargs)
        self.cip_map_path = 'data/springer/springer_second.json'
        self.load_cip_map()
        self.recv_cnt = 0

    def load_cip_map(self):
        remote_path = os.path.join('config/julian', self.cip_map_path)
        self.fetch_from_s3(remote_path, self.cip_map_path)

        with open(self.cip_map_path) as fd:
            self.cip_map = ujson.load(fd)

    def get_cip(self, p):
        return self.cip_map.get(p.replace('\\/', '/').replace(',', ''))

    def convert(self, **kwargs):
        for iid, predict in zip(
            kwargs.get('id'),\
            kwargs.get('predict')):
            # each predict in structure [iid, l1, l2]
            iid = iid.decode()
            predict = predict[2].decode()
            cip = self.get_cip(predict)

            if cip:
                cip = list(cip.keys())[0]
                yield {
                    'id': iid,
                    'cip': cip,
                }

    def send(self, **kwargs):
        self.recv_cnt += 1
        iid = kwargs.get('id')
        cip = kwargs['cip']

        try:
            table, _ = JournalArticle.rebuild(id=iid)
            brk = get_brk(JournalArticle.ID)
            res = self.get_result(cip)
            if not res:
                return

            print('++ [classi] {} {}'.format(res, self.recv_cnt))
            brk._update({JournalArticle.ID.value: iid,}, res)
            print('++ [updated] {} {}'.format(iid, cip))
        except Exception as ex:
            print('-- [error] send failed iid:{} code:{}: {}'.format(iid, cip, ex))

    def get_result(self, raw_code):
        brk_dict = get_brk(TechClassi.ID)
        cla = brk_dict._query(Key(TechClassi.ID.value).eq(
            TechClassi.make_id(raw_code))).get('Items')

        if not cla:
            print('-- [error] TechClassi items not found for code:{}'.format(raw_code))
            return

        cla = cla[0]
        return {'tech_domain_classifications': [{
                TechClassi.ID.value: raw_code,
                "name": cla['name'],
                }]}
