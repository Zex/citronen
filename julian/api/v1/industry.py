# Industry prediction API
# Author: Zex Li <top_zlynch@yahoo.com>
from flask import Flask, request, jsonify
from flask_restful import Resource
from julian.handler.naics_handler import NaicsHandler, MODE


class Industry(Resource):
    
    handler = NaicsHandler(MODE.COMPAT)
        
    def post(self):
        data = request.json

        if not data:
            return jsonify({'error': 'description not given'})

        descs = data.get('descs')
        res = [] 
        if not descs:
            return jsonify({'error': 'description not given'})

        try:
            if isinstance(descs, str):
                df = Industry.handler.predict([descs])
            else:
                df = Industry.handler.predict(descs)
            print(df['iid'].values)
            print(df['code'].values)
            list(map(lambda iid, code: res.append({'iid':iid, 'code':code}),\
                df['iid'].values, df['code'].values))
        except Exception as ex:
            print("Exception on handling request {}: {}".format(data, ex))

        return jsonify({'predicts':res})
