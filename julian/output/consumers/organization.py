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
from src.dynamodb.tables.organization import OrganizationTable
from src.dynamodb.tables.industry_classi import IndustryClassi


class Org(PC):
    """Prediction consumer for organization"""
    def __init__(self, **kwargs):
        super(Org, self).__init__(topics=(Topic.PREDICT_NAICS,), **kwargs)
        self.d3_path = "data/naics/codes_3digits.csv"
        self.d6_path = "data/naics/codes_6digits.csv"
        self.load_naics_tables()
        self.recv_cnt = 0

    def load_naics_tables(self):
        self.fetch_from_s3(os.path.join('config/julian', self.d3_path), self.d3_path)
        self.fetch_from_s3(os.path.join('config/julian', self.d6_path), self.d6_path)

        self.d3_table = pd.read_csv(self.d3_path, header=0, delimiter='#', \
                dtype={'code': np.str})
        self.d6_table = pd.read_csv(self.d6_path, header=0, delimiter='#', \
                dtype={'code': np.str})

    def get_d6_code(self, in_id):
        # TODO Single code
        codes = self.d6_table[\
            self.d6_table['code'].str[:3]==str(in_id)].values.tolist()
        return codes[np.random.randint(len(codes))][0] if codes else None

    def convert(self, **kwargs):
        """Internal conversion"""
        for iid, predict in zip(
            kwargs.get('id'),\
            kwargs.get('predict')):
            # each predict in structure [iid, d3, name]
            iid = iid.decode()
            predict = str(predict[1])
            d6_code = self.get_d6_code(predict)

            if d6_code:
                yield {
                    'id': iid,
                    'd6code': str(d6_code),
                }

    def send(self, **kwargs):
        self.recv_cnt += 1
        iid = kwargs.get('id')
        d6_code = kwargs['d6code']

        try:
            brk = get_brk(OrganizationTable.ID)
            res = self.get_result(d6_code)
            if not res:
                return

            print('++ [classi] {} {}'.format(res, self.recv_cnt))
            brk._update({OrganizationTable.ID.value: iid,}, res)
            print('++ [updated] {} {}'.format(iid, d6_code))
        except Exception as ex:
            print('-- [error] send failed iid:{} code:{}: {}'.format(iid, d6_code, ex))

    def get_result(self, raw_code):
        brk_dict = get_brk(IndustryClassi.ID)
        cla = brk_dict._query(Key(IndustryClassi.ID.value).eq(
            IndustryClassi.make_id(raw_code))).get('Items')

        if not cla:
            print('-- [error] IndustryClassi items not found for code:{}'.format(raw_code))
            return

        cla = cla[0]
        return {'industry_classifications': [{
                IndustryClassi.ID.value: raw_code,
                "name": cla['name'],
                }]}
