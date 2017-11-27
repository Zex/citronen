# Tech domain prediction API
# Author: Zex Li <top_zlynch@yahoo.com>
from flask import Flask, request, jsonify
from flask_restful import Resource
from julian.handler.springer_handler import SpringerHandler, MODE


class TechDomain(Resource):
    #TODO add log
    handler = SpringerHandler(MODE.COMPAT)
        
    def get(self, descs):
        #data = request.json
        if not descs:
            return jsonify({'error': 'description not given'})

        try:
            if isinstance(descs, str):
                res = TechDomain.handler.predict([data])
            else:
                res = TechDomain.handler.predict(data)
        except Exception as ex:
            print("Exception on handling request {}: {}".format(descs, ex))

        return jsonify({'predicts':res})
