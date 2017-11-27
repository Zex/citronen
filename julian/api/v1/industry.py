# Industry prediction API
# Author: Zex Li <top_zlynch@yahoo.com>
from flask import Flask, request, jsonify
from flask_restful import Resource
from julian.handler.naics_handler import NaicsHandler, MODE


class Industry(Resource):
    
    def __init__(self):
        self.handler = NaicsHandler(MODE.COMPAT)
        
    def get(self, descs):
        if not descs:
            return jsonify({'error': 'description not given'})

        try:
            if isinstance(descs, str):
                res = self.handler.predict([data])
            else:
                res = self.handler.predict(data)
        except Exception as ex:
            print("Exception on handling request {}: {}".format(descs, ex))

        return jsonify({'predicts':res})
