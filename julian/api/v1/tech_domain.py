# Tech domain prediction API
# Author: Zex Li <top_zlynch@yahoo.com>
from flask import Flask, request, jsonify
from flask_restful import Resource
from julian.handler.springer_handler import SpringerHandler, MODE


class TechDomain(Resource):
    #TODO add log
    handler = SpringerHandler(MODE.COMPAT)
        
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
                df = TechDomain.handler.predict([descs])
            else:
                df = TechDomain.handler.predict(descs)
            list(map(lambda iid, l1, l2: res.append({'iid':iid, 'l1':l1, 'l2':l2}),\
                df['iid'].values, df['l1'].values, df['l2'].values))
        except Exception as ex:
            print("Exception on handling request {}: {}".format(data, ex))
            raise

        return jsonify({'predicts':res})
